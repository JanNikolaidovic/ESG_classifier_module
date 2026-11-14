"""Step 1: download IFC's official list of investment projects.

Source: World Bank "Finances One" open data, dataset DS00499 "IFC Investment Services
Projects". One record per project, including its environmental category and the address
of its public disclosure page.
"""

import json

API_URL = "https://datacatalogapi.worldbank.org/dexapps/fone/api/apiservice"
API_PARAMS = {"datasetId": "DS00499", "resourceId": "RS00048", "type": "json"}
PAGE_SIZE = 1000

# IFC's category names -> our labels. Other values ("FI-1", "FI-2", "FI", "Other") are loans
# to banks or funds that lend the money on, which is not comparable to a single project.
CATEGORIES = {"A - Significant": "A", "B - Limited": "B", "C - No Impact": "C"}


def download_project_list(fetcher):
    """Return every project record, requesting the list PAGE_SIZE records at a time."""
    records, total = [], None
    while total is None or len(records) < total:
        params = {**API_PARAMS, "top": PAGE_SIZE, "skip": len(records)}
        response = json.loads(fetcher.get(API_URL, params=params))
        total = response["count"]
        if not response["data"]:
            break
        records.extend(response["data"])
    return records


def single_projects(records):
    """Keep single projects with category A, B or C, adding a short `category` field."""
    return [
        {**record, "category": CATEGORIES[record["environmental_category"]]}
        for record in records
        if record.get("environmental_category") in CATEGORIES and record.get("project_url")
    ]
