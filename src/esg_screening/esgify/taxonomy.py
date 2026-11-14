"""ESGify class metadata: E/S/G pillar and severity weight for each of the 47 labels.

Pillars follow the official ESGify class diagram (ESGify_classes.jpg in the ESGify repo).

Severity weights are a starting assumption, to be reviewed with ESG specialists and,
ideally, checked against real outcomes. The reasoning:
- HIGH: risks that typically make a project high risk in project finance (the IFC
  Performance Standards 5, 7 and 8 on resettlement, indigenous peoples and cultural
  heritage), plus severe labour, safety, pollution and integrity risks.
- MEDIUM: other environmental, social and governance impacts.
- LOW: management-system topics. Neutral descriptive text ("an independent firm will
  monitor...") often triggers these, and they rarely indicate a risk by themselves.
- NONE: benefits rather than risks (a renewable project scores ~1.0 on Energy Efficiency
  and Renewables) and the non-ESG class.
"""

ENVIRONMENTAL, SOCIAL, GOVERNANCE = "E", "S", "G"

HIGH, MEDIUM, LOW, NONE = 1.0, 0.6, 0.3, 0.0

NOT_RELEVANT = "Not Relevant to ESG"

# label: (pillar, severity weight)
LABELS = {
    # Environmental
    "Biodiversity": (ENVIRONMENTAL, HIGH),
    "Emergencies (Environmental)": (ENVIRONMENTAL, HIGH),
    "Hazardous Materials Management": (ENVIRONMENTAL, HIGH),
    "Land Acquisition and Resettlement (E)": (ENVIRONMENTAL, HIGH),
    "Soil and Groundwater Impact": (ENVIRONMENTAL, HIGH),
    "Surface Water Pollution": (ENVIRONMENTAL, HIGH),
    "Air Pollution": (ENVIRONMENTAL, MEDIUM),
    "Animal Welfare": (ENVIRONMENTAL, MEDIUM),
    "Climate Risks": (ENVIRONMENTAL, MEDIUM),
    "Greenhouse Gas Emissions": (ENVIRONMENTAL, MEDIUM),
    "Land Rehabilitation": (ENVIRONMENTAL, MEDIUM),
    "Landscape Transformation": (ENVIRONMENTAL, MEDIUM),
    "Natural Resources": (ENVIRONMENTAL, MEDIUM),
    "Physical Impacts": (ENVIRONMENTAL, MEDIUM),
    "Planning Limitations": (ENVIRONMENTAL, MEDIUM),
    "Supply Chain (Environmental)": (ENVIRONMENTAL, MEDIUM),
    "Waste Management": (ENVIRONMENTAL, MEDIUM),
    "Wastewater Management": (ENVIRONMENTAL, MEDIUM),
    "Water Consumption": (ENVIRONMENTAL, MEDIUM),
    "Environmental Management": (ENVIRONMENTAL, LOW),
    "Energy Efficiency and Renewables": (ENVIRONMENTAL, NONE),
    # Social
    "Communities Health and Safety": (SOCIAL, HIGH),
    "Cultural Heritage": (SOCIAL, HIGH),
    "Emergencies (Social)": (SOCIAL, HIGH),
    "Employee Health and Safety": (SOCIAL, HIGH),
    "Forced Labour": (SOCIAL, HIGH),
    "Human Rights": (SOCIAL, HIGH),
    "Indigenous People": (SOCIAL, HIGH),
    "Land Acquisition and Resettlement (S)": (SOCIAL, HIGH),
    "Minimum Age and Child Labour": (SOCIAL, HIGH),
    "Data Safety": (SOCIAL, MEDIUM),
    "Discrimination": (SOCIAL, MEDIUM),
    "Freedom of Association and Right to Organise": (SOCIAL, MEDIUM),
    "Labor Relations Management": (SOCIAL, MEDIUM),
    "Product Safety and Quality": (SOCIAL, MEDIUM),
    "Retrenchment": (SOCIAL, MEDIUM),
    "Supply Chain (Social)": (SOCIAL, MEDIUM),
    # Governance
    "Economic Crime": (GOVERNANCE, HIGH),
    "Legal Proceedings & Law Violations": (GOVERNANCE, HIGH),
    "Corporate Governance": (GOVERNANCE, MEDIUM),
    "Responsible Investment & Greenwashing": (GOVERNANCE, MEDIUM),
    "Supply Chain (Economic / Governance)": (GOVERNANCE, MEDIUM),
    "Values and Ethics": (GOVERNANCE, MEDIUM),
    "Disclosure": (GOVERNANCE, LOW),
    "Risk Management and Internal Control": (GOVERNANCE, LOW),
    "Strategy Implementation": (GOVERNANCE, LOW),
    # Not a risk class
    NOT_RELEVANT: (None, NONE),
}

PILLARS = (ENVIRONMENTAL, SOCIAL, GOVERNANCE)


def pillar_of(label):
    return LABELS[label][0]


def weight_of(label):
    return LABELS[label][1]
