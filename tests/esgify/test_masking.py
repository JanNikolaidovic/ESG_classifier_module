import pytest


@pytest.mark.slow
def test_masks_people_organisations_places_and_dates(masker):
    [masked] = masker.mask(["John Smith, CEO of Acme Corp, visited the plant in Texas on Monday."])
    assert masked == "<PERSON> , CEO <ORG> , visited plant <GPE> <DATE> ."


@pytest.mark.slow
def test_entities_found_with_context_before_stopword_removal(masker):
    text = "In March 2023, protesters in Lagos, Nigeria accused Shell of oil spills in the Niger Delta."
    [masked] = masker.mask([text])
    assert masked == "<DATE> , protesters <GPE> , <GPE> accused <ORG> oil spills <LOC> ."


@pytest.mark.slow
def test_text_without_entities_only_loses_stopwords(masker):
    [masked] = masker.mask(["She faced employment rejection because of her gender."])
    assert masked == "faced employment rejection gender ."
