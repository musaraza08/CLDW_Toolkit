import argparse
from pathlib import Path

from analysis import ResultsAnalysis


def main():
    parser = argparse.ArgumentParser(
        description="Run pairwise analysis over results/[ID]/ folders")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing per-ID result folders")
    parser.add_argument("--analysis-dir", default="analysis",
                        help="Directory to write analysis outputs")
    args = parser.parse_args()

    analysis = ResultsAnalysis(results_dir=Path(
        args.results_dir), analysis_dir=Path(args.analysis_dir))
    out = analysis.run()
    print(f"[INFO] Analysis written to {out}")
    print(f"[INFO] Summary at {Path(args.analysis_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
