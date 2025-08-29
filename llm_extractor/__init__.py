import json
import os
import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass

import openai
import pandas as pd

try:
    import tiktoken  # optional
except Exception:
    tiktoken = None


@dataclass
class _Result:
    trajectory: List[Dict]
    route_summary: str
    confidence: str
    error: Optional[str] = None
    best_prompt: Optional[str] = None


class LLMTrajectoryExtractor:

    TRAVEL_LABELS = ["GPE", "LOC", "FAC", "ORG"]

    def __init__(
        self,
        model: str = "gpt-4o",
        max_text_length: int = 4000,  # only when chunking is disabled
        delay: float = 0.8,
        api_key: Optional[str] = None,
        # controls
        use_chunking: bool = True,
        target_window_tokens: int = 15000,
        overlap_tokens: int = 1000,
        # model params
        json_mode: bool = True,
        temperature: float = 0.1,
        seed: Optional[int] = 42,
        # infra
        enable_cache: bool = True,
        max_retries: int = 5,
        backoff_base: float = 1.6,
        # hints
        use_prior_hints: bool = True,
    ) -> None:
        self.model = model
        self.max_text_length = max_text_length
        self.delay = delay
        self.temperature = temperature
        self.json_mode = json_mode
        self.seed = seed
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # chunking
        self.use_chunking = use_chunking
        self.target_window_tokens = target_window_tokens
        self.overlap_tokens = overlap_tokens

        # hints from prior pipeline (if available in record)
        self.use_prior_hints = use_prior_hints

        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY env variable missing and api_key not provided.")
        self.client = openai.OpenAI(api_key=api_key)

        self._cache: Dict[str, Dict] = {}

    # Public API

    def extract_for_row(self, row: pd.Series) -> Dict:
        text = str(row.get("text", ""))
        author = str(row.get("Author", ""))
        title = str(row.get("Title_short", ""))
        year = str(row.get("Year_Pub", ""))

        merged = self._extract_core(text, author, title, year)
        return {
            "ID": row.get("ID"),
            "Author": author,
            "Title": title,
            "Year": year,
            **merged,  # contains only trajectory, route_summary, confidence
        }

    def extract_record(self, record: Dict) -> Dict:
        author = record.get("Author") or record.get("author") or ""
        title = record.get("Title_short") or record.get("title") or ""
        year = record.get("Year_Pub") or record.get("year") or ""
        geoparsed = record.get("geoparsed", {}) or {}
        text = geoparsed.get("text") or record.get("text") or ""

        prior_hints = None
        if self.use_prior_hints:
            prior_hints = record.get("places") or geoparsed.get("places")

        merged = self._extract_core(
            text=str(text),
            author=str(author),
            title=str(title),
            year=str(year),
            prior_hints=prior_hints,
        )
        return {
            "ID": record.get("ID"),
            "Author": author,
            "Title": title,
            "Year": year,
            **merged,
        }

    def process_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        return [self.extract_for_row(r) for _, r in df.iterrows()]

    # Core extraction flow

    def _extract_core(
        self,
        text: str,
        author: str,
        title: str,
        year: str,
        prior_hints: Optional[List[str]] = None,
    ) -> Dict:
        # If chunking disabled, fall back to single pass
        if not self.use_chunking:
            if len(text) > self.max_text_length:
                text = text[: self.max_text_length] + "..."
            results = self._run_single_prompt(
                text, author, title, year, prior_hints)
            return self._normalize_output(results)

        # prompt per chunk, then merge
        chunks = self._chunk_text(
            text, self.target_window_tokens, self.overlap_tokens)
        per_chunk_outputs: List[Dict] = []
        for ch in chunks:
            res = self._run_single_prompt(ch, author, title, year, prior_hints)
            per_chunk_outputs.append(self._normalize_output(res))
            time.sleep(self.delay)

        merged = self._merge_pass(per_chunk_outputs)
        return merged

    def _run_single_prompt(
        self, text: str, author: str, title: str, year: str, prior_hints: Optional[List[str]]
    ) -> Dict:
        prompt = self._build_prompt_single(
            text, author, title, year, prior_hints)
        return self._run_prompt(prompt)

    # Prompt

    def _build_prompt_single(
        self, text: str, author: str, title: str, year: str, prior_hints: Optional[List[str]]
    ) -> str:
        hint_block = ""
        if prior_hints:
            uniq = ", ".join(sorted({str(h) for h in prior_hints if h}))
            if uniq:
                hint_block = (
                    "Helpful place hints (not mandatory; use only if clearly supported by the passage): "
                    + uniq
                    + "\n\n"
                )

        schema_block = (
            "Return only JSON with this exact schema and keys:\n"
            "{\n"
            '  "trajectory": [\n'
            "    {\n"
            '      "place": "string",\n'
            '      "order": "integer (1..N in chronological order)",\n'
            '      "mention": "string (verbatim phrase from the passage if possible)",\n'
            '      "note": "string (brief description of what happened/movement context)"\n'
            "    }\n"
            "  ],\n"
            '  "route_summary": "string (e.g., \"Keswick → Grasmere → Ambleside\")",\n'
            '  "confidence": "high" | "medium" | "low"\n'
            "}\n"
        )

        rules_block = (
            "Rules:\n"
            "- Identify a travel TRAJECTORY: a sequence of distinct places indicating movement.\n"
            "- Use chronologically increasing 'order' starting at 1 with no gaps.\n"
            "- Prefer canonical/standard toponyms; merge spelling variants into one canonical name.\n"
            "- Quote an exact short phrase for 'mention' when available (e.g., “set out from Keswick”).\n"
            "- Be conservative: exclude figurative, hypothetical, or purely scenic place mentions without movement.\n"
            "- Remove duplicates and near-repeats; keep the clearest single instance per place in sequence.\n"
            "- If ambiguous or only lists places without movement cues, return an empty trajectory and set confidence='low'.\n"
            "- If no clear trajectory, set route_summary='No clear travel trajectory identified'.\n"
            "- Keep the JSON minimal; do NOT include extra keys or commentary.\n"
            "- Output must be valid JSON and UTF-8. No markdown fences.\n"
            "- Maximum trajectory length: 30 items.\n"
        )

        tiny_example = (
            "Tiny example (for format only):\n"
            '{\n'
            '  "trajectory": [\n'
            '    {"place": "Keswick", "order": 1, "mention": "set out from Keswick", "note": "departure"},\n'
            '    {"place": "Grasmere", "order": 2, "mention": "passed through Grasmere", "note": "en route"},\n'
            '    {"place": "Ambleside", "order": 3, "mention": "arrived at Ambleside", "note": "arrival"}\n'
            "  ],\n"
            '  "route_summary": "Keswick → Grasmere → Ambleside",\n'
            '  "confidence": "medium"\n'
            "}\n"
        )

        header = (
            "Task: Extract a precise and succinct travel trajectory from the passage below.\n"
            f"Author: {author} | Title: {title} | Year: {year}\n\n"
        )

        return header + schema_block + rules_block + tiny_example + hint_block + "Passage:\n" + text

    # Model usage

    def _cache_key(self, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(prompt.encode())
        return h.hexdigest()

    def _run_prompt(self, prompt: str) -> Dict:
        key = None
        if self.enable_cache:
            key = self._cache_key(prompt)
            if key in self._cache:
                return self._cache[key]

        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract travel trajectories from historical texts. Respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=900,
                )
                if self.json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.seed is not None:
                    kwargs["seed"] = self.seed

                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content.strip()
                data = json.loads(content)
                if self.enable_cache and key is not None:
                    self._cache[key] = data
                return data
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "trajectory": [],
                        "route_summary": str(e),
                        "confidence": "low",
                        "error": str(e),
                    }
                time.sleep((self.backoff_base ** attempt) + 0.1)

    # Merging & Normalization

    def _normalize_output(self, result: Dict) -> Dict:
        """Ensure minimally valid shape and strip extras."""
        if not isinstance(result, dict):
            return {"trajectory": [], "route_summary": "No valid result", "confidence": "low"}
        traj = result.get("trajectory") or []
        rs = result.get("route_summary") or ""
        conf = result.get("confidence") or "low"
        # re-enforce integer 1..N order
        out_traj = []
        seen = set()
        for item in traj:
            if not isinstance(item, dict):
                continue
            place = (item.get("place") or "").strip()
            if not place:
                continue
            key = place.lower()
            if key in seen:
                continue
            seen.add(key)
            out_traj.append({
                "place": place,
                "order": int(item.get("order", len(out_traj) + 1)),
                "mention": item.get("mention", ""),
                "note": item.get("note", ""),
            })
        for i, it in enumerate(out_traj, 1):
            it["order"] = i
        return {"trajectory": out_traj, "route_summary": rs, "confidence": conf}

    def _merge_pass(self, per_chunk: List[Dict]) -> Dict:
        # Preparing compact input for the merge prompt
        chunks_json = json.dumps([c.get("trajectory", [])
                                 for c in per_chunk], ensure_ascii=False)
        prompt = (
            "You will merge multiple partial trajectories into one coherent, chronologically ordered itinerary.\n"
            "Return only JSON with this schema:\n"
            "{\n"
            '  "trajectory": [\n'
            "    {\n"
            '      "place": "string",\n'
            '      "order": "integer (1..N)",\n'
            '      "mention": "string",\n'
            '      "note": "string"\n'
            "    }\n"
            "  ],\n"
            '  "route_summary": "string",\n'
            '  "confidence": "high" | "medium" | "low"\n'
            "}\n\n"
            "Here are per-chunk trajectories (JSON array of arrays):\n"
            f"{chunks_json}\n\n"
            "Rules:\n"
            "- Keep only plausible movements; remove duplicates.\n"
            "- If two names are variants of the same place, prefer the most standard form.\n"
            "- Ensure 'order' is strictly increasing from 1..N starting at 1.\n"
            "- Keep a concise 'route_summary' like 'Keswick → Grasmere → Ambleside'.\n"
            "- Output valid JSON only, no commentary.\n"
        )
        merged = self._run_prompt(prompt)
        return self._normalize_output(merged)

    # Chunking utilities

    def _est_tokens(self, text: str) -> int:
        if tiktoken:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except Exception:
                pass
        return max(1, int(len(text) / 4))

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', (text or "").strip())
        return [s for s in sents if s]

    def _chunk_text(self, text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
        sents = self._split_sentences(text)
        chunks, cur, cur_tok = [], [], 0
        for s in sents:
            t = self._est_tokens(s)
            if cur_tok + t > target_tokens and cur:
                chunk_text = " ".join(cur)
                chunks.append(chunk_text)
                # Build overlap window from tail
                overlap, tok_sum = [], 0
                for ss in reversed(cur):
                    tok_sum += self._est_tokens(ss)
                    overlap.append(ss)
                    if tok_sum >= overlap_tokens:
                        break
                cur = list(reversed(overlap)) + [s]
                cur_tok = sum(self._est_tokens(x) for x in cur)
            else:
                cur.append(s)
                cur_tok += t
        if cur:
            chunks.append(" ".join(cur))
        return chunks
