"""Prompts for text-to-image diffusion."""

from src import hierarchy


def clean_lemma(lemma_name: str) -> str:
    """Splits lemma into separate words and makes them lowercase."""
    return " ".join(lemma_name.lower().split("_"))


def get_article(lemma_name: str) -> str:
    """Returns the article of a lemma."""
    return "an" if lemma_name[0] in "aeiou" else "a"


def base_prompt(synset: hierarchy.Synset) -> str:
    """Generates the baseline prompt for a synset."""
    lemma_name = clean_lemma(synset.lemma_names()[0])
    article = get_article(lemma_name)
    return f"An image of {article} {lemma_name}."


def all_lemmas_prompt(synset: hierarchy.Synset) -> str:
    """Generates a prompt with all lemma names."""
    if len(synset.lemma_names()) == 1:
        return base_prompt(synset)

    lemma_names = [clean_lemma(lemma_name)
                   for lemma_name in synset.lemma_names()]
    lemma_name = lemma_names[0]
    article = get_article(lemma_name)
    other_lemmas = ", ".join(lemma_names[1:])
    return f"An image of {article} {lemma_name} ({other_lemmas})."


def definition_prompt(synset: hierarchy.Synset) -> str:
    """Generates a prompt with the synset definition."""
    lemma_name = clean_lemma(synset.lemma_names()[0])
    article = get_article(lemma_name)
    return f"An image of {article} {lemma_name} ({synset.definition()})."


def photo_prompt(synset: hierarchy.Synset) -> str:
    """Generates the photo prompt for a synset (from CLIP)."""
    lemma_name = clean_lemma(synset.lemma_names()[0])
    return f"a photo of a {lemma_name}."


def painting_prompt(synset: hierarchy.Synset) -> str:
    """Generates the painting prompt for a synset (from CLIP)."""
    lemma_name = clean_lemma(synset.lemma_names()[0])
    return f"a painting of a {lemma_name}."


def empty_prompt(synset: hierarchy.Synset) -> str:
    """Generates the empty prompt for a synset (from CLIP)."""
    lemma_name = clean_lemma(synset.lemma_names()[0])
    return f"a {lemma_name}."
