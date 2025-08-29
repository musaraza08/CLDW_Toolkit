import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from validator import TrajectoryValidator


class ResultsAnalysis:
    def __init__(self,
                 results_dir: Path = Path("results"),
                 analysis_dir: Path = Path("analysis")) -> None:
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.validator = TrajectoryValidator(results_dir=self.results_dir)

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_validation_or_compute(self, rec_id: str) -> Optional[Dict[str, Any]]:
        vpath = self.results_dir / rec_id / "validation.json"
        if vpath.exists():
            return self._load_json(vpath)
        # Fallback
        llm_file = self.results_dir / rec_id / "llm_results.json"
        nlp_file = self.results_dir / rec_id / "nlp_results.json"
        if not llm_file.exists() or not nlp_file.exists():
            return None
        rec: Dict[str, Any] = {"ID": rec_id}
        rec["llm_result"] = self._load_json(llm_file) or {}
        rec["nlp_result"] = self._load_json(nlp_file) or {}
        # Populate pairwise. GT sections will be None without manual GT
        try:
            self.validator.validate_record(rec)
            return self._load_json(vpath)
        except Exception:
            return None

    def _avg(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        vals = [x for x in [a, b] if isinstance(x, (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _extract_pairwise_metrics(self, validation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pw = validation.get("pairwise") if isinstance(
            validation, dict) else None
        if not pw:
            return None
        llm_pred = pw.get("llm_as_pred")
        nlp_pred = pw.get("nlp_as_pred")
        if not llm_pred or not nlp_pred:
            return None

        def pull(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
            nodes = (m.get("metrics") or {}).get("nodes") or {}
            edges = (m.get("metrics") or {}).get("edges") or {}
            seq = (m.get("metrics") or {}).get("sequence") or {}
            return {
                "node_precision": nodes.get("precision"),
                "node_recall": nodes.get("recall"),
                "node_f1": nodes.get("f1"),
                "edge_precision": edges.get("precision"),
                "edge_recall": edges.get("recall"),
                "edge_f1": edges.get("f1"),
                "lcs_ratio": seq.get("lcs_ratio"),
                "edit_distance_norm": seq.get("edit_distance_norm"),
                "kendall_tau": seq.get("kendall_tau"),
            }

        lm = pull(llm_pred)
        nm = pull(nlp_pred)

        return {
            "node_precision_avg": self._avg(lm["node_precision"], nm["node_precision"]),
            "node_recall_avg": self._avg(lm["node_recall"], nm["node_recall"]),
            "node_f1_avg": self._avg(lm["node_f1"], nm["node_f1"]),
            "edge_precision_avg": self._avg(lm["edge_precision"], nm["edge_precision"]),
            "edge_recall_avg": self._avg(lm["edge_recall"], nm["edge_recall"]),
            "edge_f1_avg": self._avg(lm["edge_f1"], nm["edge_f1"]),
            "lcs_ratio_avg": self._avg(lm["lcs_ratio"], nm["lcs_ratio"]),
            "edit_distance_norm_avg": self._avg(lm["edit_distance_norm"], nm["edit_distance_norm"]),
            "kendall_tau_avg": self._avg(lm["kendall_tau"], nm["kendall_tau"]),
        }

    def _extract_gt_metrics(self, validation: Dict[str, Any], method_key: str) -> Optional[Dict[str, Any]]:
        sec = validation.get(method_key)
        if not isinstance(sec, dict):
            return None
        metrics = sec.get("metrics") or {}
        nodes = metrics.get("nodes") or {}
        edges = metrics.get("edges") or {}
        seq = metrics.get("sequence") or {}
        geo = sec.get("geo") or {}
        counts = sec.get("counts") or {}
        return {
            "node_precision": nodes.get("precision"),
            "node_recall": nodes.get("recall"),
            "node_f1": nodes.get("f1"),
            "edge_precision": edges.get("precision"),
            "edge_recall": edges.get("recall"),
            "edge_f1": edges.get("f1"),
            "lcs_ratio": seq.get("lcs_ratio"),
            "edit_distance_norm": seq.get("edit_distance_norm"),
            "kendall_tau": seq.get("kendall_tau"),
            "geo_mean_km": geo.get("mean_km"),
            "geo_median_km": geo.get("median_km"),
            "geo_within_1km": geo.get("pct_within_1km"),
            "geo_within_5km": geo.get("pct_within_5km"),
            "geo_within_10km": geo.get("pct_within_10km"),
            "pred_len": counts.get("pred_len"),
            "gt_len": counts.get("gt_len"),
        }

    def _collect_match_reasons(self, validation: Dict[str, Any], method_key: str) -> List[Dict[str, Any]]:
        sec = validation.get(method_key)
        if not isinstance(sec, dict):
            return []
        rows: List[Dict[str, Any]] = []
        for m in sec.get("matches", []) or []:
            reason = (m.get("reason") or "unknown").split(" ")[0]
            rows.append({
                "reason": reason
            })
        return rows

    def _plot_hist(self, series: pd.Series, title: str, fname: str, bins: int = 20) -> None:
        if not _HAS_MPL:
            return
        try:
            plt.figure(figsize=(6, 4))
            series.dropna().hist(bins=bins)
            plt.title(title)
            plt.xlabel(title)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.analysis_dir / fname, dpi=150)
            plt.close()
        except Exception:
            pass

    def _plot_box(self, series: pd.Series, title: str, fname: str) -> None:
        if not _HAS_MPL:
            return
        try:
            plt.figure(figsize=(4, 5))
            plt.boxplot(series.dropna(), vert=True)
            plt.title(title)
            plt.ylabel(title)
            plt.tight_layout()
            plt.savefig(self.analysis_dir / fname, dpi=150)
            plt.close()
        except Exception:
            pass

    def _plot_bar(self, counts: Dict[str, int], title: str, fname: str) -> None:
        if not _HAS_MPL:
            return
        try:
            items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            labels = [k for k, _ in items]
            values = [v for _, v in items]
            plt.figure(figsize=(6, 4))
            plt.bar(labels, values)
            plt.title(title)
            plt.ylabel("Count")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / fname, dpi=150)
            plt.close()
        except Exception:
            pass

    def run(self) -> Path:
        rows: List[Dict[str, Any]] = []
        ids: List[str] = []
        with_gt = 0
        gt_rows: List[Dict[str, Any]] = []
        gt_reason_rows: List[Dict[str, Any]] = []
        for d in sorted([p for p in self.results_dir.iterdir() if p.is_dir()]):
            rec_id = d.name
            validation = self._load_validation_or_compute(rec_id)
            if not validation:
                continue
            ids.append(rec_id)
            if validation.get("llm") is not None or validation.get("nlp") is not None:
                with_gt += 1
            metrics = self._extract_pairwise_metrics(validation)
            if metrics:
                metrics_row = {"ID": rec_id, **metrics}
                rows.append(metrics_row)

            # GT metrics per method (if available)
            for method_key, method_label in (("llm", "LLM"), ("nlp", "NLP")):
                gtm = self._extract_gt_metrics(validation, method_key)
                if gtm:
                    gt_rows.append(
                        {"ID": rec_id, "method": method_label, **gtm})
                    for r in self._collect_match_reasons(validation, method_key):
                        gt_reason_rows.append(
                            {"ID": rec_id, "method": method_label, **r})

        # Write pairwise CSV (agreement)
        if not rows:
            df_empty = pd.DataFrame(columns=[
                "ID", "node_precision_avg", "node_recall_avg", "node_f1_avg",
                "edge_precision_avg", "edge_recall_avg", "edge_f1_avg",
                "lcs_ratio_avg", "edit_distance_norm_avg", "kendall_tau_avg"
            ])
            csv_path = self.analysis_dir / "pairwise_metrics.csv"
            df_empty.to_csv(csv_path, index=False)
        else:
            df = pd.DataFrame(rows)
            csv_path = self.analysis_dir / "pairwise_metrics.csv"
            df.to_csv(csv_path, index=False)
            # Plots
            self._plot_hist(
                df["node_f1_avg"], "Pairwise Node F1 (avg)", "pairwise_node_f1_hist.png")
            self._plot_hist(
                df["edge_f1_avg"], "Pairwise Edge F1 (avg)", "pairwise_edge_f1_hist.png")
            self._plot_hist(df["lcs_ratio_avg"],
                            "LCS Ratio (avg)", "lcs_ratio_hist.png")
            self._plot_hist(df["kendall_tau_avg"],
                            "Kendall Tau (avg)", "kendall_tau_hist.png")

        # Writing GT metrics
        if gt_rows:
            df_gt = pd.DataFrame(gt_rows)
            df_gt_path = self.analysis_dir / "gt_metrics.csv"
            df_gt.to_csv(df_gt_path, index=False)
            # Plots
            try:
                llm_df = df_gt[df_gt["method"] == "LLM"]
                nlp_df = df_gt[df_gt["method"] == "NLP"]
                for col in ["node_f1", "edge_f1", "lcs_ratio", "kendall_tau"]:
                    if col in llm_df:
                        self._plot_hist(
                            llm_df[col], f"LLM vs GT – {col}", f"llm_vs_gt_{col}_hist.png")
                        self._plot_box(
                            llm_df[col], f"LLM vs GT – {col}", f"llm_vs_gt_{col}_box.png")
                    if col in nlp_df:
                        self._plot_hist(
                            nlp_df[col], f"NLP vs GT – {col}", f"nlp_vs_gt_{col}_hist.png")
                        self._plot_box(
                            nlp_df[col], f"NLP vs GT – {col}", f"nlp_vs_gt_{col}_box.png")
            except Exception:
                pass

        # Match reasons
        if gt_reason_rows:
            df_reason = pd.DataFrame(gt_reason_rows)
            df_reason_path = self.analysis_dir / "gt_match_reasons.csv"
            df_reason.to_csv(df_reason_path, index=False)
            # Aggregate counts per method
            try:
                llm_counts = df_reason[df_reason["method"] ==
                                       "LLM"]["reason"].value_counts().to_dict()
                nlp_counts = df_reason[df_reason["method"] ==
                                       "NLP"]["reason"].value_counts().to_dict()
                self._plot_bar(llm_counts, "Match Reasons (LLM vs GT)",
                               "llm_vs_gt_match_reasons.png")
                self._plot_bar(nlp_counts, "Match Reasons (NLP vs GT)",
                               "nlp_vs_gt_match_reasons.png")
            except Exception:
                pass

        # Summary stats
        summary = {
            "num_ids": len(ids),
            "num_with_gt": with_gt,
        }
        if rows:
            summary.update({
                "pairwise_means": pd.DataFrame(rows).drop(columns=["ID"], errors="ignore").mean(numeric_only=True).to_dict(),
                "pairwise_medians": pd.DataFrame(rows).drop(columns=["ID"], errors="ignore").median(numeric_only=True).to_dict(),
            })
        if gt_rows:
            summary.update({
                "gt_means": pd.DataFrame(gt_rows).drop(columns=["ID", "method"], errors="ignore").mean(numeric_only=True).to_dict(),
                "gt_medians": pd.DataFrame(gt_rows).drop(columns=["ID", "method"], errors="ignore").median(numeric_only=True).to_dict(),
            })
        (self.analysis_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Return a primary artifact path
        return self.analysis_dir / "pairwise_metrics.csv"


__all__ = ["ResultsAnalysis"]
