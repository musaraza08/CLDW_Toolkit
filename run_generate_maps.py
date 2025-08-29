from pathlib import Path
import json
from typing import Dict, Any

from visualizer import TrajectoryVisualizer


def load_result(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    results_root = Path("results")
    viz = TrajectoryVisualizer()

    total_dirs = 0
    generated = 0

    for rec_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        total_dirs += 1
        rec_id = rec_dir.name
        record: Dict[str, Any] = {"ID": rec_id}

        llm_file = rec_dir / "llm_results.json"
        nlp_file = rec_dir / "nlp_results.json"
        if llm_file.exists():
            record["llm_result"] = load_result(llm_file)
        if nlp_file.exists():
            record["nlp_result"] = load_result(nlp_file)

        saved_maps = viz.save_all_maps_for_record(record, rec_dir)
        if any(saved_maps.values()):
            generated += 1
            for name, p in saved_maps.items():
                if p:
                    print(f"[OK] {rec_id}: saved {name} -> {p}")
        else:
            print(f"[SKIP] {rec_id}: no trajectories available to plot")

    print(
        f"Done. Processed {total_dirs} directories; generated maps for {generated}.")


if __name__ == "__main__":
    main()
