"""Tests for the IFC dataset builder, with small hand-made pages instead of the real website."""

import json

from esg_screening.ifc.dataset import clean_description
from esg_screening.ifc.pages import description_and_rationale, extract_sections, is_rationale
from esg_screening.ifc.project_list import download_project_list, single_projects

PAGE = """
<div class="ifc_disclosure_project_content"><div class="cmp-text">
  <h2>Project Description</h2><p>The project will build a 50 MW wind farm.</p>
</div></div>
<div class="cmp-accordion__item">
  <span class="cmp-accordion__title">Location of Project and Description of Site</span>
  <div class="cmp-accordion__panel"><p>The site is farmland near the coast.</p></div>
</div>
<div class="cmp-accordion__item">
  <span class="cmp-accordion__title">Environmental &amp; Social Categorization Rationale</span>
  <div class="cmp-accordion__panel"><p>This is a Category B project.</p></div>
</div>
<div class="cmp-accordion__item">
  <span class="cmp-accordion__title">Disclaimer</span>
  <div class="cmp-accordion__panel"><p>Not relevant.</p></div>
</div>
"""


def test_extract_sections():
    sections = extract_sections(PAGE)
    assert sections["Project Description"] == "The project will build a 50 MW wind farm."
    assert set(sections) == {"Project Description", "Location of Project and Description of Site",
                             "Environmental & Social Categorization Rationale", "Disclaimer"}


def test_description_uses_only_input_sections():
    description, rationale = description_and_rationale(extract_sections(PAGE))
    assert description == "The project will build a 50 MW wind farm.\n\nThe site is farmland near the coast."
    assert rationale == "This is a Category B project."


FLAT_PAGE = """
<div class="ifc_disclosure_project_content"><div class="cmp-text"><h2>Project Description</h2>
<p>Summary Of Project Information (SPI) Project NameKenya - Clinic Company NameClinic Ltd
Technical Partner and/or Major Shareholders Dr. Smith owns 60%.
Description of Company and Purpose of Project The project will build a 40 bed clinic.
Location of Project and Description of Site The site is in Nairobi.
Environmental Category and Issues This is a category B project. Issues include medical waste.
Host country location of environmental documents Nairobi library</p>
</div></div>
"""


def test_old_one_block_page_is_split_into_sections():
    description, rationale = description_and_rationale(extract_sections(FLAT_PAGE))
    assert description == "The project will build a 40 bed clinic.\n\nThe site is in Nairobi."
    assert rationale == "This is a category B project. Issues include medical waste."


def test_old_page_variants_in_capitals_are_recognised():
    page = (FLAT_PAGE.replace("Summary Of Project Information", "SUMMARY OF\tPROJECT INFORMATION")
            .replace("Location of Project", "Location of project"))
    description, _ = description_and_rationale(extract_sections(page))
    assert description == "The project will build a 40 bed clinic.\n\nThe site is in Nairobi."


def test_rationale_titles_of_both_page_versions():
    assert is_rationale("Environmental & Social Categorization Rationale")
    assert is_rationale("Environmental and Social Issues - Category")
    assert not is_rationale("Project Description")


def test_category_mentions_removed():
    text = "The plant makes cement. This is a Category A project. It employs 200 people."
    assert clean_description(text) == "The plant makes cement. It employs 200 people."


def test_ifc_review_text_cut_off():
    text = ("The plant makes cement. Key environmental issues associated with this project include dust. "
            "An ESAP was agreed.")
    assert clean_description(text) == "The plant makes cement."
    assert clean_description("A hotel.\nThe ESRS disclosed in 2019 continues to apply.") == "A hotel."
    assert clean_description("Environmental review: none needed.") == ""


def test_old_page_with_preface_is_recognised():
    page = FLAT_PAGE.replace("Summary Of Project Information", "Please find attached - 2 - REVISED S UMMARY O F P ROJECT I NFORMATION")
    description, _ = description_and_rationale(extract_sections(page))
    assert description == "The project will build a 40 bed clinic.\n\nThe site is in Nairobi."


def test_single_projects_keeps_only_a_b_c():
    records = [
        {"environmental_category": "A - Significant", "project_url": "u1"},
        {"environmental_category": "FI-2 - Limited", "project_url": "u2"},
        {"environmental_category": "C - No Impact", "project_url": "u3"},
        {"environmental_category": "B - Limited", "project_url": None},
    ]
    assert [p["category"] for p in single_projects(records)] == ["A", "C"]


class FakeFetcher:
    """Pretends to be the API: serves 5 records, at most `top` per request."""

    def __init__(self):
        self.records = [{"project_number": str(i)} for i in range(5)]
        self.calls = 0

    def get(self, url, params):
        self.calls += 1
        page = self.records[params["skip"]:params["skip"] + params["top"]]
        return json.dumps({"count": len(self.records), "data": page})


def test_project_list_downloads_all_pages(monkeypatch):
    monkeypatch.setattr("esg_screening.ifc.project_list.PAGE_SIZE", 2)
    fetcher = FakeFetcher()
    records = download_project_list(fetcher)
    assert [r["project_number"] for r in records] == ["0", "1", "2", "3", "4"]
    assert fetcher.calls == 3
