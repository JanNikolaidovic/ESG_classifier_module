"""Step 2: read the sections of an IFC project disclosure page.

Each page has titled sections ("Project Description", "Location of Project and Description
of Site", "Environmental & Social Categorization Rationale", ...). We split the page into
{title: text} and pick:
- the INPUT sections: what a project owner would also write about their project
- the RATIONALE section: IFC's explanation of the category. It states the answer
  ("This is a Category A project..."), so it must never be used as model input.
"""

import re

from bs4 import BeautifulSoup

INPUT_SECTIONS = ("Project Description", "Location of Project and Description of Site")

# Some old pages (1990s-2000s) put the whole summary in one block, with the section titles
# inline. The block starts (within its first few words) with this header, written with varying
# capitals and spacing, e.g. "REVISED SUMMARY OF PROJECT INFORMATION" or "S UMMARY O F P ROJECT".
FLAT_PAGE_MARKER = re.compile(r"s\s?ummary\s+o\s?f\s+p\s?roject\s+i\s?nformation", re.IGNORECASE)
# The inline section titles, in their usual order; we split the block on them.
FLAT_PAGE_HEADINGS = (
    "Technical Partner and/or Major Shareholders",
    "Project Cost Including proposed IFC investment",
    "Location of Project and Description of Site",
    "Description of Company and Purpose of Project",
    "Development Impact",
    "Environmental Category and Issues",
    "Host country location of environmental documents",
    "Date SPI sent to InfoShop",
    "For Additional Information contact",
)
FLAT_HEADING_PATTERN = re.compile("(" + "|".join(re.escape(h) for h in FLAT_PAGE_HEADINGS) + ")", re.IGNORECASE)
# Capitalisation varies between pages, so titles are matched in lower case and given one spelling
STANDARD_TITLES = {h.lower(): h for h in FLAT_PAGE_HEADINGS}
STANDARD_TITLES["description of company and purpose of project"] = "Project Description"


def extract_sections(html):
    """Return {section title: section text} for an IFC project page (any page version)."""
    sections = _titled_sections(html)
    if is_flat_page(sections.get("Project Description", "")):
        sections.update(split_flat_page(sections.pop("Project Description")))
    return sections


def is_flat_page(text):
    """True if the old one-block header appears within the first 15 words of the text."""
    return bool(FLAT_PAGE_MARKER.search(" ".join(text.split()[:15])))


def split_flat_page(text):
    """Split an old one-block summary into {title: text} using its inline titles."""
    parts = FLAT_HEADING_PATTERN.split(text)  # [before, title1, text1, title2, text2, ...]
    sections = {}
    for title, body in zip(parts[1::2], parts[2::2]):
        sections[STANDARD_TITLES[title.lower()]] = body.strip(" :\n\t")
    return sections


def _titled_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    sections = {}
    # Main boxes: <div class="cmp-text"><h2>Title</h2> text ...</div>
    for box in soup.select("div.ifc_disclosure_project_content div.cmp-text"):
        heading = box.find(["h2", "h3"])
        if heading:
            title = heading.get_text(" ", strip=True)
            heading.extract()
            sections[title] = box.get_text(" ", strip=True)
    # Fold-out sections: title in span.cmp-accordion__title, text in div.cmp-accordion__panel
    for item in soup.select("div.cmp-accordion__item"):
        title = item.select_one(".cmp-accordion__title")
        panel = item.select_one(".cmp-accordion__panel")
        if title and panel:
            sections[title.get_text(" ", strip=True)] = panel.get_text(" ", strip=True)
    return sections


def is_rationale(title):
    """True for the section in which IFC explains the category (titles vary between page versions)."""
    lowered = title.lower()
    return ("categorization rationale" in lowered
            or lowered.startswith(("environmental and social issues", "environmental category")))


def description_and_rationale(sections):
    """Return (model input text, category rationale text) from a page's sections."""
    description = "\n\n".join(sections[title] for title in INPUT_SECTIONS if sections.get(title))
    rationale = "\n\n".join(text for title, text in sections.items() if is_rationale(title))
    return description, rationale
