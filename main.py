"""Command-line entry point. Run `python main.py --help` for the list of commands.

    python main.py score-esg data/samples/hydropower_project.txt   # score one description
    python main.py ifc-download                                    # 1. download IFC pages (slow)
    python main.py ifc-build                                       # 2. build the IFC dataset
    python main.py ifc-features                                    # 3. ESGify scores per IFC project (slow)
    python main.py ifc-evaluate                                    # 4. compare approaches, write report
"""

import argparse
import json
import logging
from pathlib import Path


def score_esg(args):
    from esg_screening.esgify import ESGRiskScorer  # imported here: loading flair/torch is slow

    text = Path(args.file).read_text(encoding="utf-8")
    result = ESGRiskScorer.load().score(text)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"\n{args.file}: {len(text.split())} words, {result.n_sentences} sentences, {result.n_chunks} chunks")
    print(f"\nESG risk score: {result.esg_risk_score:.1f} / 100   (risk load {result.risk_load:.2f})")
    print("  " + "   ".join(f"{p}: {s:.1f}" for p, s in result.pillar_scores.items()))
    print(f"\nFlagged risks ({len(result.flagged_risks)}):")
    for risk in result.flagged_risks[:args.top]:
        print(f"  [{risk.pillar}] {risk.label:<40} score {risk.score:.2f}  weight {risk.weight}"
              f"  -> {risk.contribution:.2f}")
        print(f"      \"{risk.evidence[:150]}{'...' if len(risk.evidence) > 150 else ''}\"")


def ifc_download(args):
    from esg_screening.ifc.dataset import download_all
    from esg_screening.ifc.fetching import PoliteFetcher

    download_all(PoliteFetcher(delay=args.delay), limit=args.limit)


def ifc_build(args):
    from esg_screening.ifc.dataset import DATASET_PATH, build_dataset

    table = build_dataset()
    print(f"\nSaved {len(table)} projects to {DATASET_PATH}")
    print(table["category"].value_counts().to_string())


def ifc_features(args):
    from esg_screening.categorizer.features import compute_features

    compute_features(limit=args.limit)


def ifc_evaluate(args):
    from esg_screening.categorizer.evaluate import run_evaluation

    run_evaluation()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="ESG screening tools")
    commands = parser.add_subparsers(dest="command", required=True)

    cmd = commands.add_parser("score-esg", help="ESG risk score of a project description text file")
    cmd.add_argument("file", help="path to a .txt project description")
    cmd.add_argument("--json", action="store_true", help="print the full result as JSON")
    cmd.add_argument("--top", type=int, default=10, help="number of flagged risks to show")
    cmd.set_defaults(func=score_esg)

    cmd = commands.add_parser("ifc-download", help="download the IFC project list and project pages")
    cmd.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1)")
    cmd.add_argument("--limit", type=int, help="download at most this many pages (for testing)")
    cmd.set_defaults(func=ifc_download)

    cmd = commands.add_parser("ifc-build", help="build data/processed/ifc_projects.csv from the downloaded pages")
    cmd.set_defaults(func=ifc_build)

    cmd = commands.add_parser("ifc-features", help="compute ESGify scores for every IFC project")
    cmd.add_argument("--limit", type=int, help="process at most this many new projects (for testing)")
    cmd.set_defaults(func=ifc_features)

    cmd = commands.add_parser("ifc-evaluate", help="compare category prediction approaches")
    cmd.set_defaults(func=ifc_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
