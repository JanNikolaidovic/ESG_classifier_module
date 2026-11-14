# ESG Screening Tool

This tool reads the **free-text description of a project** (for example, a new factory, mine or
solar farm) and tells you about its **environmental and social (E&S) risk**:

1. **Which ESG risks the text talks about**, and the sentence where each one appears.
2. **Which risk category the project most likely belongs to**: A (significant), B (limited) or
   C (minimal). This is the same A/B/C system that development banks such as the IFC use.

Descriptions must be in English.

---

## Contents

1. [Quick start](#quick-start)
2. [The big picture](#the-big-picture)
3. [Part 1: the ESG risk finder](#part-1-the-esg-risk-finder)
4. [Part 2: learning the A/B/C category from IFC projects](#part-2-learning-the-abc-category-from-ifc-projects)
5. [Results so far](#results-so-far)
6. [Known limitations](#known-limitations)
7. [Files at a glance](#files-at-a-glance)
8. [Tests](#tests)
9. [Glossary](#glossary)

---

## Quick start

```bash
# one-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# find ESG risks in one description
python main.py score-esg data/samples/hydropower_project.txt

# build the IFC dataset and measure how well we predict A/B/C (each step can be resumed)
python main.py ifc-download    # ~70 min: downloads ~4,000 public IFC project pages
python main.py ifc-build       # ~1 min: turns the pages into one table
python main.py ifc-features    # ~1.5 hours on a laptop CPU: runs Part 1 on every IFC project
python main.py ifc-evaluate    # ~1 min: compares approaches, writes reports/ifc_category_evaluation.md
```

The first run downloads two AI models into `~/.cache` (about 2 GB in total).

---

## The big picture

```
                         Project description (text)
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
     Part 1     │  ESG risk finder (uses the ESGify model) │
                │  "Which ESG risk topics are mentioned,   │
                │   where, and how strongly?"              │
                └──────────────────────────────────────────┘
                                     │  47 topic scores + evidence sentences
                                     ▼
                ┌──────────────────────────────────────────┐
     Part 2     │  Category predictor                      │
                │  learned from ~4,000 real IFC projects   │
                │  "Is this an A, B or C project?"         │
                └──────────────────────────────────────────┘
                                     │
                                     ▼
                        Category A / B / C (+ probabilities)
```

---

## Part 1: the ESG risk finder

**Command:** `python main.py score-esg <file.txt>`
**Code:** `src/esg_screening/esgify/`

It works in six small steps:

### 1. Split into sentences (`chunking.py`)
The text is cut into sentences: "The dam will flood 1,150 ha. 340 households will be relocated."
becomes two sentences.

### 2. Hide names (`masking.py`)
Names of people, companies, places and dates are replaced by tags:
"Shell spilled oil in Nigeria in 2023" → "`<ORG>` spilled oil `<GPE>` `<DATE>`".
**Why:** ESGify was trained on text with names hidden, so that it judges *what happened*, not
*who it happened to*. A second AI model (FLAIR, a "named entity recognition" model) finds the
names. Very common words like "the" and "of" are also removed, as in ESGify's training.

### 3. Group sentences into windows (`chunking.py`)
Sentences are grouped in overlapping groups of three: sentences 1-2-3, then 2-3-4, then 3-4-5, ...
**Why:** we tested the alternatives on example descriptions:
- the whole text at once: the model averages everything and specific risks get lost;
- one sentence at a time: too little context, which causes false alarms;
- three sentences: the best balance.

### 4. Score each window with ESGify (`classifier.py`)
[ESGify](https://github.com/sb-ai-lab/ESGify) is an AI model that gives every piece of text a
score from 0 to 1 for each of **47 ESG topics** (46 risk topics such as "Biodiversity",
"Forced Labour" or "Economic Crime", plus "Not Relevant to ESG").

### 5. Combine into one result (`scoring.py`, `taxonomy.py`)
- For each topic we keep its **highest score in any window**: a risk counts if it is raised
  anywhere in the text.
- Scores below 0.8 are ignored, because ESGify gives 0.5-0.8 to many unrelated topics.
- Each topic has a **severity weight** (`taxonomy.py`): 1.0 for severe risks (resettlement of
  people, indigenous peoples, forced labour, corruption...), 0.6 for other impacts, 0.3 for
  management topics, 0 for benefits such as renewable energy.
- The weighted results are added up and turned into a **0-100 risk score**, plus separate
  E, S and G scores.

### 6. Show the evidence (`pipeline.py`)
For every risk found, the tool shows the single sentence that triggered it, so a person can
check it.

**Speed:** the models take ~20 s to load, then a few seconds per description.
`ESGRiskScorer.score_many()` processes many descriptions together, which is much faster.

---

## Part 2: learning the A/B/C category from IFC projects

### Why we need real examples
The 0-100 score from Part 1 uses weights that we chose by hand. To know whether a description
belongs to a high-risk project, we need **real projects whose risk category was decided by
experts**. The **IFC** (International Finance Corporation, part of the World Bank Group)
publishes, for every project it finances, a public summary and its E&S category:

| Category | IFC's meaning | Example |
|---|---|---|
| **A** | significant, varied or irreversible impacts | a mine, a large dam |
| **B** | limited impacts that can be managed | a hotel, a food factory |
| **C** | minimal or no impacts | a software company |

### Step 1: download the list of projects (`ifc/project_list.py`)
The World Bank's open data service ("Finances One", dataset DS00499) gives a list of all ~7,200
IFC investment projects with their category and the link to their public page. We keep the
~4,000 single projects with category A, B or C. (Categories "FI-1/2/3" are loans to banks that
lend the money on, which is a different situation.)

### Step 2: download each project page politely (`ifc/fetching.py`, `ifc/dataset.py`)
The `PoliteFetcher` is careful not to burden IFC's website:
- **one page at a time**, with at least 1 second between requests;
- it **checks the site's `robots.txt`** (the file where a website says what may be downloaded);
- on a temporary error it waits longer and longer before trying again (5 s, 10 s, 20 s...);
- it says who it is (a clear "User-Agent") instead of pretending to be a browser.

Pages are saved (compressed, ~64 MB in total) in `data/raw/ifc/pages/`. Pages already saved are
skipped, so the download can be stopped and restarted at any time, for example after the
computer goes to sleep. In our run, 2 of 4,019 pages could not be downloaded (IFC's server
refuses those two addresses); they are simply skipped.

### Step 3: read the right parts of each page (`ifc/pages.py`)
Each IFC page has sections. We use only what a **project owner would also write**:

| Section | Used? | Why |
|---|---|---|
| Project Description | **yes, model input** | describes the project |
| Location of Project and Description of Site | **yes, model input** | describes where it is |
| E&S Categorization Rationale | kept aside, **never** input | it states the answer ("This is a Category A project because...") |
| Everything else (IFC's role, disclaimer, ...) | no | not about the project itself |

Using the rationale as input would be **leakage**: the model would learn to read the answer
instead of judging the project, and would look excellent in tests but be useless in practice.

Old pages (1990s-2000s) need extra care, because they put the whole summary in one block,
including IFC's environmental review. Three safety measures (`pages.py`, `dataset.py`):
1. **Old one-block pages are split** on their inline titles ("Description of Company and Purpose
   of Project", "Environmental Category and Issues", ...), so they get the same sections as new pages.
2. **IFC's own review is cut off.** In IFC's texts the review always comes after the description,
   so the text is cut at the start of the first sentence that begins it ("Key environmental issues
   associated with this project include...", "...as defined in the ESRS", "E&S due diligence").
3. **Any sentence naming a category** ("This is a Category B project") is removed.

A check over all descriptions for these words ("category A/B/C", "environmental review",
"ESRS", "Summary of Project Information") finds zero matches after cleaning.

### Step 4: build one table (`ifc/dataset.py`)
`python main.py ifc-build` writes `data/processed/ifc_projects.csv`: one row per project with
its description, category, industry, country, year and rationale. Descriptions shorter than 20
words and duplicate descriptions are dropped.

Result of our run: **3,994 projects: 306 A, 3,168 B, 520 C**, approved between 1993 and 2026.
Descriptions are 148 words long on average (median), up to ~2,000.

### Step 5: run Part 1 on every IFC project (`categorizer/features.py`)
`python main.py ifc-features` runs the ESG risk finder on each description and saves the 47
topic scores (plus the hand-made 0-100 score) to `data/processed/ifc_esgify_features.csv`.
- It saves every 25 projects, so it can be stopped and resumed.
- It processes projects in **random order**, so the projects finished so far are always a fair
  sample (useful for an early look at results).
- Each row stores a fingerprint of the description: if a description changes, it is scored again.

### Step 6: measure how well we predict the category (`categorizer/evaluate.py`)
`python main.py ifc-evaluate` compares several approaches, all tested in exactly the same way:

- **Approach (a), our main candidate:** the 47 ESGify topic scores go into a *logistic
  regression*, a simple model that learns one weight per topic from the IFC examples. So the
  weights are learned from real decisions instead of chosen by hand.
- **Comparisons** that tell us whether (a) is any good:
  - "Always B": always guess the most common category. Anything useful must beat this.
  - "Industry only": guess from the industry alone (e.g. mining → A), without reading the text.
  - "Hand-made ESG risk score": our 0-100 score from Part 1.
  - "(a) + industry": approach (a) plus the industry.
  - "Word counts (TF-IDF)": a classic text method that only counts words, without ESGify.

**How the test works (5-fold cross-validation):** the projects are split into 5 groups. Each
group in turn is hidden; the approach learns from the other 4 groups and predicts the hidden
one. Every project is predicted once by a model that never saw it. Projects of the same company
always stay in the same group, so an approach can't score well by recognising a company.

**Rare category A:** only ~8% of projects are A. The models are told to treat mistakes on A
as more important ("balanced class weights"), because missing an A project is the costly error.

The full report is written to `reports/ifc_category_evaluation.md`.

---

## Results so far

Full report: [`reports/ifc_category_evaluation.md`](reports/ifc_category_evaluation.md).
Test set: all 3,994 IFC projects (306 A, 3,168 B, 520 C), with 5-fold cross-validation.

| Approach | A ranking (ROC AUC) | Balanced accuracy | Flags that are real A, when 80% of A are caught |
|---|---|---|---|
| Always B (guessing) | 0.50 | 0.33 | 8% (1 in 12) |
| Hand-made ESG risk score (Part 1) | 0.58 | 0.43 | 8% |
| Industry only (no text) | 0.77 | 0.70 | 12% |
| **(a) ESGify 47 topic scores** | **0.77** | **0.67** | **14% (1 in 7)** |
| (a) + industry | 0.81 | 0.71 | 17% |
| **Word counts (TF-IDF)** | **0.90** | **0.76** | **27% (about 1 in 4)** |

On our owner-style examples, word counts gave the expected answer for all four (dairy B, hydropower
A, both rooftop solar projects B). Approach (a) wrongly called the solar project that lists the risks
it does *not* have an A project.

**What this means, in plain words:**
- **IFC's category depends mostly on what kind of project it is, how big it is and where.** The words
  that push most towards A are *power, construction, km, transmission, gas, mining, pipeline*; towards
  C: *bank, financial, services, network*. Plain word counts capture this directly.
- **ESGify's topic scores are only a weak, indirect signal for the category**: no better than knowing
  the industry. Some of what approach (a) learned is even counter-intuitive (e.g. "Forced Labour" and
  "Human Rights" push *away* from A), which confirms that ESGify measures which topics a text talks
  about, not how risky the project is.
- **The hand-made 0-100 score from Part 1 barely predicts the category** (0.58, where 0.5 is random).
  It should be read as "how much ESG-risk language the text contains", not as a risk level.
- **Current recommendation:** predict the category with a text classifier (word counts now; a fine-tuned
  language model is the next thing to try), and keep ESGify for what it does well: pointing to the
  sentences that mention each ESG risk topic, as an explanation for a human reviewer.

---

## Known limitations

- **ESGify detects topics, not whether a risk exists.** It was trained on news, where
  mentioning a risk usually means something went wrong. In project descriptions, "no
  resettlement is needed" raises the same topic as "300 families will be resettled". A test
  with sentences saying a risk is *present*, *mitigated* or *absent* showed similar scores for
  all three. Part 2 learns from real categories, which reduces but does not remove this problem.
- **IFC descriptions are written by IFC staff, not by project owners.** Their style is more
  neutral and factual than an owner's description, so results on owner texts may differ. The
  examples in `data/samples/` are owner-style texts used to check this.
- **Weights and thresholds in Part 1** were set by hand on a few examples, and the resulting 0-100
  score barely predicts IFC's category (see [Results so far](#results-so-far)).
- **Only four owner-style test descriptions** exist so far (`data/samples/`); results on real owner
  texts still need checking.
- **Speed:** a few seconds per description on a laptop CPU.

---

## Files at a glance

```
main.py                         command line: score-esg, ifc-download, ifc-build, ifc-features, ifc-evaluate
src/esg_screening/
    esgify/                     Part 1: ESG risk finder
        classifier.py             the ESGify model; scores texts in batches
        masking.py                hides names (FLAIR model)
        chunking.py               sentences and 3-sentence windows
        taxonomy.py               topic -> E/S/G pillar and severity weight
        scoring.py                combines window scores into one result and a 0-100 score
        pipeline.py               ESGRiskScorer: runs all steps; score() and score_many()
        nltk_data.py              downloads small language files the first time
    ifc/                        Part 2a: IFC dataset builder
        project_list.py           step 1: list of IFC projects (World Bank open data)
        fetching.py               step 2: polite downloader
        pages.py                  step 3: reads the right sections of a page
        dataset.py                steps 2+4: downloads all pages, builds the table
    categorizer/                Part 2b: predicting A/B/C
        features.py               step 5: Part 1 scores for every IFC project
        evaluate.py               step 6: compares approaches, writes the report
data/samples/                   example owner-style descriptions (low, medium, high risk, and
                                one low-risk project that lists the risks it does NOT have)
data/raw/, data/processed/      downloaded and generated data (not in git; rebuild with main.py)
reports/                        evaluation reports
tests/                          automatic tests
```

---

## Tests

```bash
pytest                  # everything (~1-2 min; loads the AI models)
pytest -m "not slow"    # quick tests only (under a second, no AI models, no internet)
```

The quick tests use small hand-made examples (fake web pages, fake model scores) so they check
our own logic. The slow tests run the real models on the example descriptions.

---

## Glossary

- **ESG**: Environmental, Social and Governance, the three groups of non-financial risk.
- **E&S category (A/B/C)**: how serious a project's environmental and social impacts could be,
  as decided by a development bank before financing it.
- **IFC**: International Finance Corporation, the World Bank Group's lender to private companies.
- **ESGify**: a free AI model that scores text for 47 ESG topics.
- **Named entity recognition (NER)**: finding names of people, companies, places and dates in text.
- **Feature**: a number that describes an example and is given to a model as input (here: the
  47 ESGify topic scores).
- **Logistic regression**: a simple, transparent model that learns one weight per feature and
  turns their weighted sum into probabilities for A, B and C.
- **TF-IDF**: a classic way to turn a text into numbers by counting words, giving rare words more
  importance than common ones.
- **Leakage**: when the answer accidentally ends up in the model's input, which makes test results
  look much better than reality.
- **Cross-validation**: testing a model on data it did not learn from, several times over
  different splits, so every example is used for testing once.
- **Recall / precision for A**: recall = share of real A projects that we catch; precision = share
  of our "A" predictions that are really A. Catching more A projects usually means more false alarms,
  so we also compare approaches at the same catch rate (80% of A projects caught).
- **Balanced accuracy**: the average of the share of A, B and C projects predicted correctly, so the
  rare category A counts as much as the common B. Guessing gives 0.33.
- **ROC AUC**: how well the predicted chance of "A" sorts real A projects above the others:
  0.5 = random order, 1.0 = perfect order.
