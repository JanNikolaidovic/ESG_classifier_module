# Predicting IFC's E&S category from the project description

Data: 3994 IFC projects with a description (A: 306, B: 3168, C: 520). Test method: 5-fold cross-validation, projects of the same company kept together.

## Results

| Approach | Macro F1 | Balanced accuracy | A found (recall) | A correct (precision) | A ranking (ROC AUC) | A correct when 80% of A found |
|---|---|---|---|---|---|---|
| Always B (most common category) | 0.29 | 0.33 | 0.00 | 0.00 | 0.50 | 0.08 |
| Industry only (no text) | 0.60 | 0.70 | 0.67 | 0.21 | 0.77 | 0.12 |
| Hand-made ESG risk score | 0.19 | 0.43 | 0.29 | 0.14 | 0.58 | 0.08 |
| (a) ESGify 47 class scores | 0.52 | 0.67 | 0.65 | 0.19 | 0.77 | 0.14 |
| (a) + industry | 0.58 | 0.71 | 0.67 | 0.20 | 0.81 | 0.17 |
| Word counts (TF-IDF), no ESGify | 0.73 | 0.76 | 0.56 | 0.44 | 0.90 | 0.27 |

What the columns mean (all between 0 and 1, higher is better):
- **Macro F1**: overall quality, with A, B and C counting equally however rare they are.
- **Balanced accuracy**: the average of the share of A, B and C projects predicted correctly. Guessing gives 0.33.
- **A found (recall)**: of the real Category A projects, the share predicted as A. Missing an A is the costly mistake.
- **A correct (precision)**: of the projects predicted as A, the share that really are A.
- **A ranking (ROC AUC)**: if we sort projects by predicted chance of being A, how often a real A comes before a non-A. 0.5 = random, 1 = perfect.
- **A correct when 80% of A found**: we lower the bar for flagging "A" until 80% of the real A projects are flagged; this is the share of flagged projects that really are A. It compares approaches fairly for screening, where catching most A projects matters most.

## (a) ESGify 47 class scores: in detail

Confusion matrix (rows: real category, columns: predicted):

| real \ predicted | A | B | C |
|---|---|---|---|
| A | 198 | 79 | 29 |
| B | 809 | 1819 | 540 |
| C | 20 | 87 | 413 |

Inputs that push most **towards** Category A: Landscape Transformation (+0.46), Energy Efficiency and Renewables (+0.27), Responsible Investment & Greenwashing (+0.25), Employee Health and Safety (+0.22), Supply Chain (Environmental) (+0.21), Natural Resources (+0.19), Biodiversity (+0.18), Corporate Governance (+0.18), Environmental Management (+0.16), Discrimination (+0.16)

Inputs that push most **away** from Category A: Risk Management and Internal Control (-0.34), Data Safety (-0.33), Minimum Age and Child Labour (-0.30), Forced Labour (-0.26), Human Rights (-0.25), Communities Health and Safety (-0.25), Freedom of Association and Right to Organise (-0.16), Supply Chain (Economic / Governance) (-0.16), Retrenchment (-0.15), Product Safety and Quality (-0.12)

## Word counts (TF-IDF), no ESGify: in detail

Confusion matrix (rows: real category, columns: predicted):

| real \ predicted | A | B | C |
|---|---|---|---|
| A | 171 | 130 | 5 |
| B | 209 | 2832 | 127 |
| C | 6 | 86 | 428 |

Inputs that push most **towards** Category A: power (+2.78), construction (+2.61), km (+2.04), transmission (+1.46), cocoa (+1.41), development (+1.41), gas (+1.38), toll (+1.37), mining (+1.36), pipeline (+1.34)

Inputs that push most **away** from Category A: services (-1.35), network (-1.25), financial (-1.19), market (-1.13), bank (-0.95), company (-0.94), provide (-0.90), companies (-0.89), products (-0.88), leading (-0.87)

## Our own example descriptions (owner-style text, not IFC)

Each cell: predicted category (probability of A). Industries were set by hand for these samples.

| Sample | Expected (our judgement) | Industry only (no text) | Hand-made ESG risk score | (a) ESGify 47 class scores | (a) + industry | Word counts (TF-IDF), no ESGify |
|---|---|---|---|---|---|---|
| dairy_plant_expansion.txt | B | B (0.32) | A (0.54) | B (0.23) | B (0.21) | B (0.11) |
| hydropower_project.txt | A | A (0.64) | B (0.00) | A (0.72) | A (0.75) | A (0.73) |
| rooftop_solar_project.txt | B or C | A (0.64) | C (0.28) | B (0.40) | A (0.46) | B (0.16) |
| solar_with_risk_denials.txt | B or C | A (0.64) | B (0.19) | A (0.53) | A (0.83) | B (0.17) |
