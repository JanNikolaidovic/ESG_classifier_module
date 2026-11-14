"""Command-line entry point. Run `python main.py --help` for the list of commands.

Using the tool on a project description:
    python main.py predict-category data/samples/hydropower_project.txt  # category A/B/C + reasons
    python main.py score-esg data/samples/hydropower_project.txt         # ESG risk topics only

Building it from public IFC data (once, in this order):
    python main.py ifc-download     # 1. download IFC pages (slow)
    python main.py ifc-build        # 2. build the IFC dataset
    python main.py train-category   # 3. train and save the category model
    python main.py ifc-features     # optional: ESGify scores per IFC project (slow), for ifc-evaluate
    python main.py ifc-evaluate     # optional: compare approaches, write report
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


def predict_category(args):
    from esg_screening.categorizer import model as category_model

    text = Path(args.file).read_text(encoding="utf-8")
    try:
        model, metadata = category_model.load()
    except FileNotFoundError as error:
        raise SystemExit(str(error))  # a clear message instead of a long error report
    [(category, probabilities)] = category_model.predict(model, [text])
    towards, away = category_model.explain(model, text)
    risks = []
    if not args.no_risks:
        from esg_screening.esgify import ESGRiskScorer  # slow: loads two AI models
        risks = ESGRiskScorer.load().score(text).flagged_risks[:args.top]

    if args.json:
        print(json.dumps({
            "category": category,
            "probabilities": probabilities,
            "words_towards_A": [word for word, _ in towards],
            "words_away_from_A": [word for word, _ in away],
            "risk_topics": [{"label": r.label, "pillar": r.pillar, "score": r.score, "evidence": r.evidence}
                            for r in risks],
            "model": {key: metadata[key] for key in ("model", "trained_at", "training_data", "cross_validation")},
        }, indent=2))
        return

    print(f"\n{args.file} ({len(text.split())} words)")
    print(f"\nPredicted E&S category: {category}  ({category_model.MEANINGS[category]})")
    print("  " + "   ".join(f"{c}: {probabilities[c]:.0%}" for c in category_model.CATEGORIES))
    print(f"\nWords in this text pointing towards A: {', '.join(word for word, _ in towards) or '-'}")
    print(f"Words pointing away from A:            {', '.join(word for word, _ in away) or '-'}")
    if risks:
        print("\nESG risk topics mentioned (ESGify), strongest first:")
        for risk in risks:
            print(f"  [{risk.pillar}] {risk.label}")
            print(f"      \"{risk.evidence[:150]}{'...' if len(risk.evidence) > 150 else ''}\"")
    data, tested = metadata["training_data"], metadata["cross_validation"]
    print(f"\nModel trained {metadata['trained_at'][:10]} on {data['projects']:,} IFC projects "
          f"(tested: A ranking {tested['A ranking (ROC AUC)']:.2f}, balanced accuracy {tested['Balanced accuracy']:.2f})")


def train_category(args):
    from esg_screening.categorizer.train import train_and_save

    _, metrics = train_and_save()
    print("\nCross-validated test results of the saved model:")
    for name, value in metrics.items():
        print(f"  {name:32s} {value:.2f}")


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

    cmd = commands.add_parser("predict-category", help="predict the E&S category (A/B/C) of a project description")
    cmd.add_argument("file", help="path to a .txt project description (English)")
    cmd.add_argument("--json", action="store_true", help="print the full result as JSON")
    cmd.add_argument("--top", type=int, default=8, help="number of ESG risk topics to show")
    cmd.add_argument("--no-risks", action="store_true", help="skip the ESG risk topics (much faster)")
    cmd.set_defaults(func=predict_category)

    cmd = commands.add_parser("train-category", help="train the category model on the IFC dataset and save it")
    cmd.set_defaults(func=train_category)

    cmd = commands.add_parser("score-esg", help="ESG risk topics in a project description text file")
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
