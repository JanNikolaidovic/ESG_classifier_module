"""Build the IFC training dataset: one row per project, description text + category A/B/C.

Two steps, so the slow part never has to be repeated:
1. download_all: save the project list and every project page to data/raw/ifc/ (slow,
   about one page per second; pages already saved are skipped, so it can be stopped and resumed).
2. build_dataset: read the saved pages and write data/processed/ifc_projects.csv (fast, offline).
"""

import gzip
import json
import logging
import re
from pathlib import Path

import pandas as pd

from esg_screening.ifc.pages import description_and_rationale, extract_sections
from esg_screening.ifc.project_list import download_project_list, single_projects

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/ifc")
DATASET_PATH = Path("data/processed/ifc_projects.csv")

# Sentences that name the category would give the answer away, e.g. "This is a Category B project".
CATEGORY_MENTION = re.compile(r"\bcategory\s+[ABC]\b", re.IGNORECASE)

# Signs that IFC's own environmental review starts. In IFC's texts it always comes after the
# project description, so everything from the first of these onwards is cut off.
REVIEW_START = re.compile(
    r"s\s?ummary\s+o\s?f\s+p\s?roject\s+i\s?nformation"  # an old summary page appended to the text
    r"|environmental\s+(and\s+social\s+)?(category|categori[sz]ation|issues|review)"
    r"|\bESRS\b|\bESAP\b|E\s?&\s?S\s+due\s+diligence|scope\s+of\s+IFC.s",
    re.IGNORECASE,
)


def download_all(fetcher, raw_dir=RAW_DIR, limit=None):
    """Download the project list and the page of every A/B/C project not downloaded yet."""
    pages_dir = raw_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading the project list")
    records = download_project_list(fetcher)
    (raw_dir / "project_list.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    projects = single_projects(records)
    log.info("%d projects in total, %d with category A/B/C", len(records), len(projects))

    missing = [p for p in projects if not _page_path(pages_dir, p).exists()]
    todo = missing[:limit]
    log.info("%d pages already saved, downloading %d", len(projects) - len(missing), len(todo))
    failed = 0
    for i, project in enumerate(todo, 1):
        try:
            html = fetcher.get(project["project_url"])
        except Exception as error:  # one broken page must not stop the whole download
            failed += 1
            log.warning("Skipping project %s: %s", project["project_number"], error)
            continue
        with gzip.open(_page_path(pages_dir, project), "wt", encoding="utf-8") as file:
            file.write(html)
        if i % 100 == 0 or i == len(todo):
            log.info("%d / %d pages downloaded (%d failed)", i, len(todo), failed)


def build_dataset(raw_dir=RAW_DIR, output_path=DATASET_PATH):
    """Turn the saved pages into a table and save it as CSV. Returns the table."""
    records = json.loads((raw_dir / "project_list.json").read_text(encoding="utf-8"))
    rows = []
    for project in single_projects(records):
        path = _page_path(raw_dir / "pages", project)
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as file:
            description, rationale = description_and_rationale(extract_sections(file.read()))
        rows.append({
            "project_number": project["project_number"],
            "project_name": project["project_name"],
            "company_name": project["company_name"],
            "industry": project["industry"],
            "country": project["country"],
            "year": project["board_approval___calendar_year"],
            "category": project["category"],
            "description": clean_description(description),
            "rationale": rationale,  # for reference only, never a model input
        })

    table = pd.DataFrame(rows)
    before = len(table)
    table = table[table["description"].str.split().str.len() >= 20]  # empty or near-empty pages
    table = table.drop_duplicates(subset="description")  # the same text published twice
    log.info("%d pages read, %d kept after removing empty and duplicate descriptions", before, len(table))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def clean_description(text):
    """Keep only the project description itself, so the answer can't leak into the input:
    cut off IFC's own environmental review, then drop any sentence that names a category."""
    review = REVIEW_START.search(text)
    if review:
        # cut at the start of the sentence (or line) in which the review starts
        sentence_start = max(text.rfind(". ", 0, review.start()) + 1, text.rfind("\n", 0, review.start()), 0)
        text = text[:sentence_start]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(s for s in sentences if not CATEGORY_MENTION.search(s))


def _page_path(pages_dir, project):
    return pages_dir / f"{project['project_number']}.html.gz"
