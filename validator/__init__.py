import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geopy.distance import geodesic


@dataclass
class MatchConfig:
    distance_km_threshold: float = 5.0
    fuzzy_name_threshold: float = 0.92


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if s == "" or s.lower() == "null":
            return None
        return float(s)
    except Exception:
        return None


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).lower()
    # strip punctuation and normalize whitespace
    out = []
    prev_space = False
    for ch in s:
        if ch.isalnum() or ch in [" "]:
            if ch == " ":
                if not prev_space:
                    out.append(" ")
                prev_space = True
            else:
                out.append(ch)
                prev_space = False
        # skip punctuation
    norm = "".join(out).strip()
    return norm or None


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    # Simple token-based Jaccard similarity
    at = set(_normalize_name(a).split())
    bt = set(_normalize_name(b).split())
    if not at or not bt:
        return 0.0
    inter = len(at & bt)
    union = len(at | bt)
    return inter / union if union else 0.0


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            if ai == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


def _kendall_tau(pred_gt_indices: List[int]) -> Optional[float]:
    n = len(pred_gt_indices)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred_gt_indices[i] < pred_gt_indices[j]:
                concordant += 1
            elif pred_gt_indices[i] > pred_gt_indices[j]:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None
    return (concordant - discordant) / denom


class TrajectoryValidator:
    def __init__(self,
                 manual_gt_path: Path = Path(
                     "results/manual_trajectories.json"),
                 results_dir: Path = Path("results"),
                 config: Optional[MatchConfig] = None) -> None:
        self.manual_gt_path = Path(manual_gt_path)
        self.results_dir = Path(results_dir)
        self.config = config or MatchConfig()
        self._manual_index: Dict[str, Dict[str, Any]] = {}
        self._load_manual()

    def _load_manual(self) -> None:
        with self.manual_gt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Index by id
        for rec in data:
            rid = rec.get("id") or rec.get("ID")
            if not rid:
                continue
            self._manual_index[str(rid)] = rec

    def _extract_gt(self, rec_id: str) -> List[Dict[str, Any]]:
        manual = self._manual_index.get(str(rec_id))
        if not manual:
            raise ValueError(
                f"Ground truth ID {rec_id} not found in manual trajectories")
        seq = manual.get("trajectory", [])
        return [self._coerce_place(x) for x in seq]

    def _coerce_place(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, str):
            return {
                "place": item,
                "lat": None,
                "long": None,
                "type": None,
                "gazref": None,
                "conf": None,
            }
        if isinstance(item, dict):
            return {
                "place": item.get("place"),
                "lat": _safe_float(item.get("lat")),
                "long": _safe_float(item.get("long")),
                "type": item.get("type"),
                "gazref": item.get("gazref"),
                "conf": item.get("conf"),
                "variant": item.get("variant"),
            }
        return {
            "place": None,
            "lat": None,
            "long": None,
            "type": None,
            "gazref": None,
            "conf": None,
        }

    def _coerce_pred_place(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            # unify key naming: support lon/long
            lng = item.get("long", item.get("lon"))
            return {
                "place": item.get("place") or item.get("name") or item.get("text"),
                "lat": _safe_float(item.get("lat")),
                "long": _safe_float(lng),
                "type": item.get("type"),
                "gazref": item.get("gazref"),
                "order": item.get("order"),
            }
        if isinstance(item, str):
            return {"place": item, "lat": None, "long": None, "type": None, "gazref": None}
        return {"place": None, "lat": None, "long": None, "type": None, "gazref": None}

    def _build_name_aliases(self, gt_item: Dict[str, Any]) -> List[str]:
        names = []
        if gt_item.get("place"):
            names.append(gt_item["place"])
        if gt_item.get("variant"):
            names.append(gt_item["variant"])
        return names

    def _match_nodes(self,
                     pred_seq: List[Dict[str, Any]],
                     gt_seq: List[Dict[str, Any]]) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
        used_gt: set = set()
        mapping: Dict[int, int] = {}
        match_info: List[Dict[str, Any]] = []
        for i, p in enumerate(pred_seq):
            p_name = p.get("place")
            p_norm = _normalize_name(p_name) if p_name else None
            p_lat = p.get("lat")
            p_lng = p.get("long")
            p_gaz = p.get("gazref")

            best: Optional[Tuple[int, str, float]] = None

            # Tier 1: gazref exact match
            if p_gaz:
                for j, g in enumerate(gt_seq):
                    if j in used_gt:
                        continue
                    if g.get("gazref") and g.get("gazref") == p_gaz:
                        best = (j, "gazref", 0.0)
                        break

            # Tier 2: geo proximity
            if best is None and p_lat is not None and p_lng is not None:
                nearest_j = None
                nearest_d = math.inf
                for j, g in enumerate(gt_seq):
                    if j in used_gt:
                        continue
                    g_lat = g.get("lat")
                    g_lng = g.get("long")
                    if g_lat is None or g_lng is None:
                        continue
                    try:
                        d_km = geodesic((p_lat, p_lng), (g_lat, g_lng)).km
                    except Exception:
                        continue
                    if d_km < nearest_d:
                        nearest_d = d_km
                        nearest_j = j
                if nearest_j is not None and nearest_d <= self.config.distance_km_threshold:
                    best = (nearest_j, "geo<=%.1fkm" %
                            self.config.distance_km_threshold, nearest_d)

            # Tier 3: name match
            if best is None and p_norm:
                best_sim = 0.0
                best_j = None
                for j, g in enumerate(gt_seq):
                    if j in used_gt:
                        continue
                    alias_names = self._build_name_aliases(g)
                    for alias in alias_names:
                        sim = _name_similarity(p_norm, alias)
                        if sim > best_sim:
                            best_sim = sim
                            best_j = j
                if best_j is not None and best_sim >= self.config.fuzzy_name_threshold:
                    best = (best_j, f"name>=%.2f" %
                            self.config.fuzzy_name_threshold, 1.0 - best_sim)

            if best is not None:
                mapping[i] = best[0]
                used_gt.add(best[0])
                match_info.append({
                    "pred_index": i,
                    "gt_index": best[0],
                    "pred_place": p_name,
                    "gt_place": gt_seq[best[0]].get("place"),
                    "reason": best[1],
                    "score": best[2],
                })

        return mapping, match_info

    def _edge_list(self, n: int) -> List[Tuple[int, int]]:
        return [(i, i + 1) for i in range(n - 1)] if n >= 2 else []

    def _compute_metrics(self,
                         pred_seq: List[Dict[str, Any]],
                         gt_seq: List[Dict[str, Any]],
                         mapping: Dict[int, int]) -> Dict[str, Any]:
        tp = len(mapping)
        fp = max(0, len(pred_seq) - tp)
        fn = max(0, len(gt_seq) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _f1(precision, recall)

        # Edge metrics
        pred_edges = self._edge_list(len(pred_seq))
        gt_edges = self._edge_list(len(gt_seq))
        matched_edges = 0
        matched_gt_edges_set = set()
        for (i, j) in pred_edges:
            if i in mapping and j in mapping:
                gi = mapping[i]
                gj = mapping[j]
                if gj - gi == 1:
                    matched_edges += 1
                    matched_gt_edges_set.add((gi, gj))
        edge_tp = matched_edges
        edge_fp = max(0, len(pred_edges) - edge_tp)
        edge_fn = max(0, len(gt_edges) - len(matched_gt_edges_set))
        edge_precision = edge_tp / \
            (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0.0
        edge_recall = edge_tp / \
            (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0.0
        edge_f1 = _f1(edge_precision, edge_recall)

        # Sequence metrics (name-based)
        pred_names = [_normalize_name(p.get("place")) or "" for p in pred_seq]
        gt_names = [_normalize_name(g.get("place")) or "" for g in gt_seq]
        lcs_len = _lcs_length(pred_names, gt_names)
        lcs_ratio = lcs_len / len(gt_names) if len(gt_names) > 0 else 0.0
        edit_dist = _edit_distance(pred_names, gt_names)
        edit_norm = edit_dist / \
            len(gt_names) if len(gt_names) > 0 else float(edit_dist)
        sequence_accuracy = 1.0 if pred_names == gt_names else 0.0
        n_pos = min(len(pred_names), len(gt_names))
        positional_matches = sum(1 for k in range(
            n_pos) if pred_names[k] == gt_names[k]) if n_pos > 0 else 0
        positional_accuracy = (positional_matches /
                               n_pos) if n_pos > 0 else 0.0

        # Kendall tau for matched nodes
        pred_ordered_matched_gt_indices = [
            mapping[i] for i in sorted(mapping.keys())]
        kendall_tau = _kendall_tau(pred_ordered_matched_gt_indices)

        return {
            "nodes": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            "edges": {
                "precision": edge_precision,
                "recall": edge_recall,
                "f1": edge_f1,
                "tp": edge_tp,
                "fp": edge_fp,
                "fn": edge_fn,
            },
            "sequence": {
                "lcs_ratio": lcs_ratio,
                "edit_distance": edit_dist,
                "edit_distance_norm": edit_norm,
                "sequence_accuracy": sequence_accuracy,
                "positional_accuracy": positional_accuracy,
                "kendall_tau": kendall_tau,
            },
        }

    def _geo_stats(self,
                   pred_seq: List[Dict[str, Any]],
                   gt_seq: List[Dict[str, Any]],
                   mapping: Dict[int, int]) -> Dict[str, Any]:
        distances: List[float] = []
        for pi, gi in mapping.items():
            p = pred_seq[pi]
            g = gt_seq[gi]
            if p.get("lat") is None or p.get("long") is None:
                continue
            if g.get("lat") is None or g.get("long") is None:
                continue
            try:
                d = geodesic((p["lat"], p["long"]), (g["lat"], g["long"])).km
                distances.append(d)
            except Exception:
                continue
        if not distances:
            return {
                "count": 0,
                "mean_km": None,
                "median_km": None,
                "pct_within_1km": None,
                "pct_within_5km": None,
                "pct_within_10km": None,
            }
        distances.sort()
        mean_km = statistics.mean(distances)
        median_km = statistics.median(distances)
        n = len(distances)
        within_1 = sum(1 for d in distances if d <= 1.0) / n
        within_5 = sum(1 for d in distances if d <= 5.0) / n
        within_10 = sum(1 for d in distances if d <= 10.0) / n
        return {
            "count": n,
            "mean_km": mean_km,
            "median_km": median_km,
            "pct_within_1km": within_1,
            "pct_within_5km": within_5,
            "pct_within_10km": within_10,
        }

    def _build_report(self,
                      rec_id: str,
                      method_label: str,
                      pred_seq_raw: List[Any],
                      gt_seq_raw: List[Any]) -> Dict[str, Any]:
        pred_seq = [self._coerce_pred_place(x) for x in pred_seq_raw]
        gt_seq = [self._coerce_place(x) for x in gt_seq_raw]

        # collapse consecutive duplicates by normalized name for robustness
        def collapse(seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            last = None
            for item in seq:
                nm = _normalize_name(item.get("place")) or ""
                if last is None or (_normalize_name(last.get("place")) or "") != nm:
                    out.append(item)
                    last = item
            return out

        pred_seq = collapse(pred_seq)
        gt_seq = collapse(gt_seq)

        mapping, matches = self._match_nodes(pred_seq, gt_seq)
        metrics = self._compute_metrics(pred_seq, gt_seq, mapping)
        geo = self._geo_stats(pred_seq, gt_seq, mapping)

        unmatched_pred = [i for i in range(len(pred_seq)) if i not in mapping]
        unmatched_gt = [j for j in range(
            len(gt_seq)) if j not in mapping.values()]

        # Simplified sections for reporting
        # NER-style metrics (alias of node metrics)
        ner_section = {
            "precision": metrics["nodes"].get("precision"),
            "recall": metrics["nodes"].get("recall"),
            "f1": metrics["nodes"].get("f1"),
            "tp": metrics["nodes"].get("tp"),
            "fp": metrics["nodes"].get("fp"),
            "fn": metrics["nodes"].get("fn"),
        }

        # Trajectory reconstruction metrics
        traj_seq = metrics.get("sequence", {})
        trajectory_section = {
            "edit_distance": traj_seq.get("edit_distance"),
            "edit_distance_norm": traj_seq.get("edit_distance_norm"),
            "sequence_accuracy": traj_seq.get("sequence_accuracy"),
            "positional_accuracy": traj_seq.get("positional_accuracy"),
            "geospatial": geo,
        }

        # Relations / routes metrics
        relations_section = {
            "precision": metrics["edges"].get("precision"),
            "recall": metrics["edges"].get("recall"),
            "f1": metrics["edges"].get("f1"),
            "kendall_tau": traj_seq.get("kendall_tau"),
        }

        # Error analysis
        pred_names = [_normalize_name(p.get("place")) or "" for p in pred_seq]
        gt_names = [_normalize_name(g.get("place")) or "" for g in gt_seq]
        gt_name_set = set([n for n in gt_names if n])
        hallucinations_count = sum(1 for idx in range(
            len(pred_seq)) if idx in unmatched_pred and (pred_names[idx] not in gt_name_set))

        # Inversions for mis-orderings
        inv_num = 0
        matched_idx = [mapping[i] for i in sorted(mapping.keys())]
        for i in range(len(matched_idx)):
            for j in range(i + 1, len(matched_idx)):
                if matched_idx[i] > matched_idx[j]:
                    inv_num += 1
        inv_denom = (len(matched_idx) * (len(matched_idx) - 1)) // 2
        inversion_fraction = (inv_num / inv_denom) if inv_denom > 0 else 0.0

        # Implicit inference
        def edges_from_names(names: List[str]) -> List[tuple]:
            return [(names[k], names[k + 1]) for k in range(len(names) - 1)] if len(names) >= 2 else []
        pred_edges_names = edges_from_names(pred_names)
        gt_edges_names = set(edges_from_names(gt_names))
        implicit_edges = [
            e for e in pred_edges_names if e not in gt_edges_names]
        implicit_inference_count = len(implicit_edges)
        implicit_inference_prop = (
            implicit_inference_count / len(pred_edges_names)) if len(pred_edges_names) > 0 else 0.0

        errors_section = {
            "hallucinations": {
                "count": hallucinations_count,
                "proportion_pred": (hallucinations_count / len(pred_seq)) if len(pred_seq) > 0 else 0.0,
            },
            "mis_orderings": {
                "num_inversions": inv_num,
                "inversion_fraction": inversion_fraction,
            },
            "implicit_inference": {
                "count": implicit_inference_count,
                "proportion_pred_edges": implicit_inference_prop,
            },
        }

        return {
            "id": rec_id,
            "method": method_label,
            "counts": {
                "pred_len": len(pred_seq),
                "gt_len": len(gt_seq),
            },
            "metrics": metrics,
            "geo": geo,
            "ner": ner_section,
            "trajectory_simple": trajectory_section,
            "relations_simple": relations_section,
            "errors": errors_section,
            "matches": matches,
            "unmatched_pred_indices": unmatched_pred,
            "unmatched_gt_indices": unmatched_gt,
        }

    def validate_id(self, rec_id: str) -> Path:
        rec_dir = self.results_dir / f"{rec_id}"
        llm_path = rec_dir / "llm_results.json"
        nlp_path = rec_dir / "nlp_results.json"
        if not llm_path.exists() and not nlp_path.exists():
            raise FileNotFoundError(
                f"No per-method results found in {rec_dir} (expected llm_results.json and/or nlp_results.json)")
        rec: Dict[str, Any] = {"ID": rec_id}
        if llm_path.exists():
            with llm_path.open("r", encoding="utf-8") as f:
                rec["llm_result"] = json.load(f)
        if nlp_path.exists():
            with nlp_path.open("r", encoding="utf-8") as f:
                rec["nlp_result"] = json.load(f)
        return self.validate_record(rec)

    def validate_record(self, rec: Dict[str, Any]) -> Path:
        rec_id = rec.get("ID") or rec.get("id")
        if not rec_id:
            raise ValueError("Record missing ID")

        has_gt = str(rec_id) in self._manual_index
        gt_seq = self._extract_gt(str(rec_id)) if has_gt else None
        out: Dict[str, Any] = {"ID": rec_id}

        # LLM vs GT
        if has_gt and isinstance(rec.get("llm_result"), dict) and rec["llm_result"].get("trajectory"):
            out["llm"] = self._build_report(
                # type: ignore[arg-type]
                str(rec_id), "LLM", rec["llm_result"]["trajectory"], gt_seq
            )
        else:
            out["llm"] = None

        # NLP vs GT
        if has_gt and isinstance(rec.get("nlp_result"), dict) and rec["nlp_result"].get("trajectory"):
            out["nlp"] = self._build_report(
                # type: ignore[arg-type]
                str(rec_id), "NLP", rec["nlp_result"]["trajectory"], gt_seq
            )
        else:
            out["nlp"] = None

        # Pairwise LLM vs NLP (both directions) if both exist
        pairwise = None
        if isinstance(rec.get("llm_result"), dict) and isinstance(rec.get("nlp_result"), dict):
            llm_traj = rec["llm_result"].get("trajectory") or []
            nlp_traj = rec["nlp_result"].get("trajectory") or []
            if llm_traj and nlp_traj:
                pairwise = {
                    "llm_as_pred": self._build_report(str(rec_id), "LLM_vs_NLP", llm_traj, nlp_traj),
                    "nlp_as_pred": self._build_report(str(rec_id), "NLP_vs_LLM", nlp_traj, llm_traj),
                }
        out["pairwise"] = pairwise

        # Write JSON report
        self.results_dir.mkdir(parents=True, exist_ok=True)
        rec_dir = self.results_dir / f"{rec_id}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        out_path = rec_dir / "validation.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out_path
