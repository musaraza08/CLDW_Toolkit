import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm_extractor import LLMTrajectoryExtractor
from nlp_extractor import NLPTrajectoryExtractor
from data_loader import DataLoader
from visualizer import TrajectoryVisualizer
from validator import TrajectoryValidator

PARSED_CSV = Path("data/cldw_parsed.csv")
OUTPUT_JSON = Path("data/trajectories_comparison.json")
RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser(
        description="Run both LLM and NLP trajectory extractors on the same subset of the CLDW corpus and save side-by-side results."
    )
    parser.add_argument("--num-records", type=int, default=1,
                        help="Number of records to compare (ignored if --id is set)")
    parser.add_argument(
        "--id", help="Specific metadata ID value to compare (overrides --num-records)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM extraction (useful when key missing)")
    parser.add_argument("--no-nlp", action="store_true",
                        help="Skip NLP extraction")
    parser.add_argument("--llm-model", default="gpt-4o",
                        help="OpenAI model for LLM pipeline")
    parser.add_argument("--output", default=OUTPUT_JSON,
                        help="Combined JSON output file (aggregate)")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory to store per-record JSON results and maps")
    args = parser.parse_args()

    loader = DataLoader()

    # Determine which records to process
    records_to_process = []
    if args.id:
        row_df = loader.df[loader.df["ID"] == args.id]
        if row_df.empty:
            raise ValueError(f"ID {args.id} not found in metadata CSV.")
        seq_val = int(row_df.iloc[0]["Seq"])
        records_to_process.append(loader.load_record(seq_val))
    else:
        seq_list = loader.df["Seq"].head(args.num_records).tolist()
        for seq in seq_list:
            try:
                records_to_process.append(loader.load_record(int(seq)))
            except Exception as exc:
                print(f"[WARN] Could not load record Seq {seq}: {exc}")

    # Instantiate extractors and visualizer
    llm_extractor = None if args.no_llm else LLMTrajectoryExtractor(
        model=args.llm_model)
    nlp_extractor = None if args.no_nlp else NLPTrajectoryExtractor()
    visualizer = TrajectoryVisualizer()
    validator = TrajectoryValidator(results_dir=Path(args.results_dir))

    output_records = []
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(records_to_process, desc="Comparing trajectories"):
        out_rec = {
            "ID": rec.get("ID"),
            "Author": rec.get("Author"),
            "Title": rec.get("Title_short"),
            "Year": rec.get("Year_Pub"),
        }
        if llm_extractor:
            out_rec["llm_result"] = llm_extractor.extract_record(rec)
        if nlp_extractor:
            out_rec["nlp_result"] = nlp_extractor.extract_record(rec)

        # Write per-method minimal JSON inside results/<ID>/
        rec_dir = results_dir / f"{out_rec['ID']}"
        rec_dir.mkdir(parents=True, exist_ok=True)

        if out_rec.get("llm_result"):
            llm_res = out_rec["llm_result"]
            llm_min = {
                "trajectory": llm_res.get("trajectory", []),
                "route_summary": llm_res.get("route_summary"),
                "confidence": llm_res.get("confidence"),
            }
            with (rec_dir / "llm_results.json").open("w", encoding="utf-8") as f_llm:
                json.dump(llm_min, f_llm, ensure_ascii=False, indent=2)

        if out_rec.get("nlp_result"):
            nlp_res = out_rec["nlp_result"]
            nlp_min = {
                "trajectory": nlp_res.get("trajectory", []),
                "route_summary": nlp_res.get("route_summary"),
                "confidence": nlp_res.get("confidence"),
            }
            with (rec_dir / "nlp_results.json").open("w", encoding="utf-8") as f_nlp:
                json.dump(nlp_min, f_nlp, ensure_ascii=False, indent=2)

        # Save maps: GT vs LLM, GT vs NLP, and combined (GT+LLM+NLP)
        saved_maps = visualizer.save_all_maps_for_record(out_rec, rec_dir)
        any_saved = any(saved_maps.values())
        if any_saved:
            for name, path in saved_maps.items():
                if path:
                    print(f"[INFO] Saved {name} map to {path}")
        else:
            print(f"[WARN] No trajectory data to plot for {out_rec['ID']}")

        # Validate and save validation.json inside results/<ID>/
        try:
            validator.validate_id(str(out_rec['ID']))
        except Exception as e:
            print(f"[WARN] Validation failed for {out_rec['ID']}: {e}")

        output_records.append(out_rec)

    # Write combined JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print(
        f"[INFO] Wrote combined JSON for {len(output_records)} texts to {output_path}")
    print(f"[INFO] Individual JSON files and maps saved in {results_dir}")


if __name__ == "__main__":
    main()
