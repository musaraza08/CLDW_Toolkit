from __future__ import annotations

"""
Rule-based + dependency-driven trajectory extractor for travel narratives.

Major improvements over the original version:
- Extracts structured movement EVENTS (src → dst via [vias]) per sentence/clause.
- Uses spaCy DependencyMatcher for FROM/TO/VIA instead of string searches.
- Adds negation handling, quoted-speech penalty, and vision/planning verb filtering.
- Maintains a simple state machine to chain events into a coherent path.
- Optionally uses lat/lon for plausibility (haversine distance) and large-admin filtering.
- Keeps original public API: extract_for_row(...), process_dataframe(...), extract_record(...).
"""

from typing import Dict, List, Optional, Tuple
import difflib
from math import radians, sin, cos, asin, sqrt

import pandas as pd
import spacy
from spacy.matcher import DependencyMatcher, Matcher
from tqdm import tqdm

# Defaults & Cue Lexicons
DEFAULT_MODEL = "en_core_web_lg"
DEFAULT_LABELS = ["GPE", "LOC", "FAC"]

# Movement / arrival cues (lemmas)
ARRIVE_LEMMAS = {"arrive", "reach", "come"}
MOVE_LEMMAS = {
    "go", "head", "proceed", "travel", "journey", "walk", "ride", "sail", "row",
    "ford", "turn", "leave", "depart", "pass", "cross", "visit", "approach",
    "descend", "ascend", "return", "follow", "make", "set"
}
# Vision verbs
EXCLUDE_SENSE_VERBS = {"see", "saw", "seen", "view",
                       "behold", "overlook", "glimpse", "watch", "observe"}
# Planning verbs
PLANNING_LEMMAS = {"intend", "plan", "hope", "decide"}
# Return phrases
RETURN_PHRASES = ("returned to", "back to")

# Historical / multiword cues
MULTIWORD_CUES = [
    ("set", "off"), ("set", "out"), ("made", "for"), ("went", "on"), ("go", "on"),
    ("came", "to"), ("arrived", "at"), ("put",
                                        "in"), ("made", "way"), ("took", "road"),
    ("betook", "to"), ("alight", "at"), ("repair", "to")
]

# Helpers


def _coerce_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        # Strip stray whitespace and commas
        if isinstance(x, str):
            xs = x.strip().replace(",", "")
            return float(xs)
    except Exception:
        return None
    return None


def haversine(lat1: Optional[float], lon1: Optional[float],
              lat2: Optional[float], lon2: Optional[float]) -> Optional[float]:
    lat1 = _coerce_float(lat1)
    lon1 = _coerce_float(lon1)
    lat2 = _coerce_float(lat2)
    lon2 = _coerce_float(lon2)
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def in_quotes(text: str) -> bool:
    return any(q in text for q in ('"', "“", "”", "'", "’"))


class NLPTrajectoryExtractor:

    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_LABELS = DEFAULT_LABELS

    @staticmethod
    def _load_spacy_model(model_name: str):
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise SystemExit(
                f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
            ) from exc

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 labels: Optional[List[str]] = None,
                 max_text_length: int = 100_000,
                 enable_distance_plausibility: bool = True,
                 expose_events: bool = False) -> None:
        self.labels: List[str] = labels or NLPTrajectoryExtractor.DEFAULT_LABELS
        self.max_text_length = max_text_length
        self.enable_distance_plausibility = enable_distance_plausibility
        self.expose_events = expose_events

        self.nlp = NLPTrajectoryExtractor._load_spacy_model(model)
        self.dep = self._build_dep_matchers(self.nlp, self.labels)
        self.matcher = self._build_token_matchers(self.nlp)

    # Public API

    def extract_for_row(self, row: pd.Series) -> Dict:
        txt = row["text"]
        doc = self.nlp(txt)
        trajectory, events = self._extract(doc)
        names = [p["place"] for p in trajectory]
        result = {
            "ID": row.get("ID"),
            "Author": row.get("Author"),
            "Title": row.get("Title_short"),
            "Year": row.get("Year_Pub"),
            "trajectory": trajectory,
            "route_summary": " → ".join(names) if names else "No clear travel trajectory identified",
            "confidence": self._confidence_from_events(events, trajectory),
            "method": "spaCy dep-matcher + heuristics",
        }
        if self.expose_events:
            result["events"] = events
        return result

    def process_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        return [self.extract_for_row(r) for _, r in tqdm(df.iterrows(), total=len(df))]

    def extract_record(self, record: Dict) -> Dict:
        text = ""
        gp = record.get("geoparsed")
        if gp:
            text = gp.get("text", "")
        if not text:
            text = record.get("text", "")
        doc = self.nlp(text)
        trajectory, events = self._extract(
            doc, gp.get("entities", []) if gp else None)
        names = [p["place"] for p in trajectory]
        result = {
            "ID": record.get("ID"),
            "Author": record.get("Author"),
            "Title": record.get("Title_short"),
            "Year": record.get("Year_Pub"),
            "trajectory": trajectory,
            "route_summary": " → ".join(names) if names else "No clear travel trajectory identified",
            "confidence": self._confidence_from_events(events, trajectory),
            "method": "spaCy dep-matcher + heuristics",
        }
        if self.expose_events:
            result["events"] = events
        return result

    # Internal utilities

    @staticmethod
    def _build_token_matchers(nlp) -> Matcher:
        m = Matcher(nlp.vocab)
        # Multiword cues as token sequences
        for a, b in MULTIWORD_CUES:
            m.add(f"MWC_{a}_{b}".upper(), [[{"LEMMA": a}, {"LOWER": b}]])
        return m

    @staticmethod
    def _build_dep_matchers(nlp, labels: List[str]) -> DependencyMatcher:
        dep = DependencyMatcher(nlp.vocab)
        # VERB -> prep(to|into|onto|at) -> pobj(Place)
        dep.add("TO_DEST", [[
            {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {
                "LOWER": {"IN": ["to", "into", "onto", "at"]}}},
            {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "dest",
                "RIGHT_ATTRS": {"ENT_TYPE": {"IN": labels}}},
        ]])
        # VERB -> prep(from) -> pobj(Place)
        dep.add("FROM_SRC", [[
            {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {"LOWER": "from"}},
            {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "src",
                "RIGHT_ATTRS": {"ENT_TYPE": {"IN": labels}}},
        ]])
        # VERB -> prep(via|through|by) -> pobj(Place)
        dep.add("VIA", [[
            {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {"LOWER": {"IN": ["via", "through", "by"]}}},
            {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "via",
                "RIGHT_ATTRS": {"ENT_TYPE": {"IN": labels}}},
        ]])
        return dep

    @staticmethod
    def _confidence(n_nodes: int) -> str:
        if n_nodes >= 5:
            return "high"
        if n_nodes >= 3:
            return "medium"
        return "low"

    def _confidence_from_events(self, events: List[Dict], trajectory: List[Dict]) -> str:
        if not trajectory:
            return "low"
        avg_score = sum(e.get("score", 0)
                        for e in events) / max(len(events), 1)
        # blend with path length
        if avg_score >= 2.0 and len(trajectory) >= 4:
            return "high"
        if avg_score >= 1.0 and len(trajectory) >= 2:
            return "medium"
        return self._confidence(len(trajectory))

    # Geoparsed entity indexing / normalization
    @staticmethod
    def _index_geoparsed(geoparsed_entities: Optional[List[Dict]]) -> Dict[str, Dict]:
        entity_index: Dict[str, Dict] = {}
        if geoparsed_entities:
            for e in geoparsed_entities:
                n = (e.get("name") or "").strip()
                if n:
                    entity_index[n.lower()] = e
        return entity_index

    @staticmethod
    def _is_large_admin(ent_dict: Optional[Dict]) -> bool:
        if not ent_dict:
            return False
        t = (ent_dict.get("type") or "").lower()
        return t in {"large admin unit", "rgn", "pplc"}

    @staticmethod
    def _normalize_name(name: str, entity_index: Dict[str, Dict]) -> Tuple[str, Optional[Dict]]:
        key = (name or "").strip()
        if not key:
            return name, None
        low = key.lower()
        if low in entity_index:
            ent = entity_index[low]
            return ent.get("name", key), ent
        # Fuzzy match for historical variants / typos
        candidates = difflib.get_close_matches(
            low, list(entity_index.keys()), n=1, cutoff=0.85)
        if candidates:
            ent = entity_index[candidates[0]]
            return ent.get("name", key), ent
        return key, None

    # Core extraction

    def _extract(self, doc: spacy.tokens.Doc, geoparsed_entities: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[Dict]]:
        entity_index = self._index_geoparsed(geoparsed_entities)

        def ent_meta_for_span(span: spacy.tokens.Span):
            name, meta = self._normalize_name(span.text, entity_index)
            return name, meta

        def entity_span_for_token(tok: spacy.tokens.Token, sent: spacy.tokens.Span):
            for e in sent.ents:
                if e.start <= tok.i < e.end and e.label_ in self.labels:
                    return e
            return None

        # Run dependency matcher once on doc
        dep_matches = self.dep(doc)
        dep_by_token: Dict[int, List[Tuple[str, List[int]]]] = {
            i: [] for i in range(len(doc))}
        for match_id, token_ids in dep_matches:
            rule_name = self.dep.vocab.strings[match_id]
            if not token_ids:
                continue
            v_idx = token_ids[0]  # by construction, first node is the verb
            dep_by_token[v_idx].append((rule_name, token_ids))

        events: List[Dict] = []

        for sent_i, sent in enumerate(doc.sents):
            s_start, s_end = sent.start, sent.end
            s_text = sent.text.strip()

            # Skip sentences that only discuss vision or plans
            verb_lemmas = {t.lemma_.lower() for t in sent if t.pos_ == "VERB"}
            if verb_lemmas & PLANNING_LEMMAS:
                continue
            if verb_lemmas & EXCLUDE_SENSE_VERBS:
                continue

            # Candidate verbs: movement/arrival
            verb_tokens = [t for t in sent if t.pos_ == "VERB" and (
                t.lemma_.lower() in MOVE_LEMMAS or t.lemma_.lower() in ARRIVE_LEMMAS)]
            if not verb_tokens:
                # allow multiword cue as weak signal
                mw_matches = self.matcher(doc[s_start:s_end])
                if not mw_matches:
                    continue

            # Gather dep matches anchored in this sentence
            local_dep_matches: List[Tuple[str, List[int]]] = []
            for i in range(s_start, s_end):
                for rule_name, token_ids in dep_by_token.get(i, []):
                    # ensure all nodes lie within sentence boundary
                    if all(s_start <= idx < s_end for idx in token_ids):
                        local_dep_matches.append((rule_name, token_ids))

            # Extract src/dst/via from dep patterns
            src = dst = None
            vias: List[Tuple[str, Dict]] = []
            cue_strength = 0.0
            negated = False
            v_head: Optional[spacy.tokens.Token] = None

            # Walk matches to build event
            for rule_name, token_ids in local_dep_matches:
                v = doc[token_ids[0]]
                if any(ch.dep_ == "neg" for ch in v.children):
                    negated = True
                v_head = v
                if rule_name == "TO_DEST" and len(token_ids) >= 3:
                    d_tok = doc[token_ids[2]]
                    d_span = entity_span_for_token(d_tok, sent)
                    if d_span is not None:
                        d_name, d_meta = ent_meta_for_span(d_span)
                        if d_meta and not self._is_large_admin(d_meta):
                            dst = (d_name, d_meta)
                            cue_strength += 2.0
                elif rule_name == "FROM_SRC" and len(token_ids) >= 3:
                    s_tok = doc[token_ids[2]]
                    s_span = entity_span_for_token(s_tok, sent)
                    if s_span is not None:
                        s_name, s_meta = ent_meta_for_span(s_span)
                        if s_meta and not self._is_large_admin(s_meta):
                            src = (s_name, s_meta)
                            cue_strength += 2.0
                elif rule_name == "VIA" and len(token_ids) >= 3:
                    v_tok = doc[token_ids[2]]
                    v_span = entity_span_for_token(v_tok, sent)
                    if v_span is not None:
                        v_name, v_meta = ent_meta_for_span(v_span)
                        if v_meta and not self._is_large_admin(v_meta):
                            vias.append((v_name, v_meta))
                            cue_strength += 1.0

            # Arrival verbs without explicit prep
            if not dst:
                for v in verb_tokens:
                    if v.lemma_.lower() in ARRIVE_LEMMAS and v.i >= s_start:
                        # nearest right entity in sentence
                        ents = [
                            e for e in sent.ents if e.label_ in self.labels and e.start >= v.i]
                        if ents:
                            d_name, d_meta = ent_meta_for_span(ents[0])
                            if d_meta and not self._is_large_admin(d_meta):
                                dst = (d_name, d_meta)
                                cue_strength += 1.0
                                v_head = v
                                break

            # If still nothing, weak multiword cue + single place entity → treat as dst
            if not (src or dst):
                if self.matcher(doc[s_start:s_end]):
                    ents = [e for e in sent.ents if e.label_ in self.labels]
                    if ents:
                        d_name, d_meta = ent_meta_for_span(ents[-1])
                        if d_meta and not self._is_large_admin(d_meta):
                            dst = (d_name, d_meta)
                            cue_strength += 0.5

            if not (src or dst):
                continue

            # Quote penalty
            if in_quotes(s_text):
                cue_strength *= 0.5

            events.append({
                "src": src,
                "dst": dst,
                "vias": vias,
                "verb": (v_head.lemma_.lower() if v_head else None),
                "score": cue_strength,
                "negated": bool(negated),
                "sent_id": sent_i,
                "text": s_text,
            })

        # Chain events into a path
        path: List[Dict] = []
        current: Optional[Tuple[str, Dict]] = None
        order_idx = 1
        last_added: Optional[str] = None

        for ev in sorted(events, key=lambda e: e["sent_id"]):
            if ev["negated"] or ev["score"] <= 0:
                continue

            # Optional distance plausibility filter
            if self.enable_distance_plausibility and current and ev.get("dst"):
                # Support alternate longitude key names in metadata ("long", "lon", "lng")
                def _get_lon(meta: Dict):
                    return meta.get("long", meta.get("lon", meta.get("lng")))

                def _get_lat(meta: Dict):
                    return meta.get("lat")
                d = haversine(_get_lat(current[1]), _get_lon(current[1]),
                              _get_lat(ev["dst"][1]), _get_lon(ev["dst"][1]))
                if d is not None and d > 250 and (ev["verb"] not in ARRIVE_LEMMAS and ev["verb"] not in MOVE_LEMMAS):
                    # Very long jump without a strong travel verb → drop
                    continue

            # Prefer explicit src -> dst; else treat dst as continuation
            if ev["src"] and ev["dst"]:
                s_name, s_meta = ev["src"]
                d_name, d_meta = ev["dst"]
                # add src if path empty or last place != src
                if not path or (last_added or "").lower() != s_name.lower():
                    node = self._make_node(
                        s_name, s_meta, order_idx, ev["text"])
                    path.append(node)
                    order_idx += 1
                    last_added = s_name
                # add vias before destination
                for v_name, v_meta in ev["vias"]:
                    if (last_added or "").lower() != v_name.lower():
                        node = self._make_node(
                            v_name, v_meta, order_idx, ev["text"])
                        path.append(node)
                        order_idx += 1
                        last_added = v_name
                # add destination
                if (last_added or "").lower() != d_name.lower():
                    node = self._make_node(
                        d_name, d_meta, order_idx, ev["text"])
                    path.append(node)
                    order_idx += 1
                    last_added = d_name
                current = ev["dst"]

            elif ev["dst"]:
                d_name, d_meta = ev["dst"]
                if (last_added or "").lower() != d_name.lower():
                    node = self._make_node(
                        d_name, d_meta, order_idx, ev["text"])
                    path.append(node)
                    order_idx += 1
                    last_added = d_name
                current = ev["dst"]

        # Collapsing consecutive duplicates
        collapsed: List[Dict] = []
        for node in path:
            if collapsed and collapsed[-1]["place"].lower() == node["place"].lower():
                continue
            collapsed.append(node)

        return collapsed, events

    # Node builder
    @staticmethod
    def _make_node(name: str, meta: Dict, order_idx: int, context: str) -> Dict:
        return {
            "place": name,
            "order": order_idx,
            "context": context,
            "lat": meta.get("lat"),
            "lon": meta.get("long"),
            "type": meta.get("type"),
        }


__all__ = ["NLPTrajectoryExtractor"]
