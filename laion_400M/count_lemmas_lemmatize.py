import nltk
import os
import sys
import pathlib
import json
import time
import tqdm
from urllib.request import urlretrieve
import numpy as np
import pandas as pd


class LemmaCounter:
    def __init__(self, synset_lemmas_path):
        with open(synset_lemmas_path) as synset_lemmas_file:
            self.synset_lemmas = json.load(synset_lemmas_file)

        self.all_lemmas = sorted([
            lemma.lower() for lemma in
            sum(self.synset_lemmas.values(), start=[])
        ])
        self.lemma_to_id = {
            lemma: i for i, lemma in enumerate(self.all_lemmas)
        }

    def count_lemmas(self, parquet_path: pathlib.Path, save_path: str, part_id: str):
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()

        all_texts = pd.read_parquet(parquet_path).caption.str.lower()

        lemma_counts = np.zeros(len(self.all_lemmas), dtype=np.int32)
        for text in tqdm.tqdm(all_texts, mininterval=60):
            text = text if text is not None else ""
            lemmas = set([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])
            for lemma in lemmas:
                lemma_id = self.lemma_to_id.get(lemma)
                if lemma_id is not None:
                    lemma_counts[lemma_id] += 1

        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / f"{part_id}.npy", lemma_counts)


def main():
    print(sys.argv)
    _, synset_lemmas_path, save_path, part_id, cache_folder = sys.argv

    parquet_name = f"{part_id}.parquet"
    parquet_path = f"{cache_folder}/{parquet_name}"
    url = f"https://huggingface.co/datasets/laion/laion400m/resolve/main/{parquet_name}"

    urlretrieve(url, parquet_path)

    lemma_counter = LemmaCounter(synset_lemmas_path)
    lemma_counter.count_lemmas(parquet_path, save_path, part_id)

    os.remove(parquet_path)


if __name__ == "__main__":
    main()
