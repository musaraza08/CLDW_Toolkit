import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set

from validator import TrajectoryValidator


def load_manual_ids(manual_path: Path) -> Set[str]:
    with manual_path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    ids: Set[str] = set()
    for rec in data:
        rid = rec.get("id") or rec.get("ID")
        if rid is not None:
            ids.add(str(rid))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate all results against ground truth and write validation.json per ID")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing per-ID result folders")
    parser.add_argument("--manual-gt", default="results/manual_trajectories.json",
                        help="Path to manual ground truth trajectories JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    manual_path = Path(args.manual_gt)

    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")
    if not manual_path.exists():
        raise SystemExit(f"Manual ground truth JSON not found: {manual_path}")

    gt_ids = load_manual_ids(manual_path)
    if not gt_ids:
        raise SystemExit(
            "No ground truth IDs found in manual trajectories JSON")

    validator = TrajectoryValidator(
        manual_gt_path=manual_path, results_dir=results_dir)

    ids_validated = 0
    ids_skipped = 0
    for d in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        rec_id = d.name
        if rec_id not in gt_ids:
            ids_skipped += 1
            continue
        try:
            out_path = validator.validate_id(rec_id)
            print(f"[OK] {rec_id} -> {out_path}")
            ids_validated += 1
        except Exception as e:
            print(f"[ERR] {rec_id}: {e}")

    print(f"[DONE] Validated: {ids_validated}, Skipped (no GT): {ids_skipped}")


if __name__ == "__main__":
    main()
