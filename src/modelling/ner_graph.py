import sys
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import spacy


class NerNetworkModel:
    """ """

    def __init__(self):
        if "en_core_web_sm" not in spacy.util.get_installed_models():
            try:
                # Try to download the model
                print("downloading spacy package en_core_web_sm")
                spacy.cli.download("en_core_web_sm")
            except Exception as e:
                print("Error downloading 'en_core_web_sm' model:", e)
                sys.exit(1)

        self.ner_model = spacy.load("en_core_web_sm")

    def extract_verb_triplets(
        self,
        text: List[str],
        entity_types: List[str] = ["PERSON", "NORP", "FAC", "ORG", "EVENT", "LAW"],
    ):
        triplets = []
        # Process the sentence with spaCy
        for sentence in text:
            doc = self.ner_model(sentence)

            # Extract dependency triplets involving specified entity types
            for token in doc:
                if token.dep_ == "nsubj" or token.dep_ == "dobj":
                    if token.ent_type_ in entity_types:
                        subject = token.head.text
                        relation = token.dep_
                        object_ = token.text
                        triplets.append((subject, relation, object_))

        return triplets
