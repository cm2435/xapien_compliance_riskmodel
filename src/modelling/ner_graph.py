import sys
from typing import List, Tuple
import spacy


class NerNetworkModel:
    """
    A class for extracting verb triplets using spaCy's named entity recognition (NER) model.

    Attributes:
        ner_model (spacy.Language): The loaded spaCy NER model.
    """

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
    ) -> List[Tuple[str, str, str]]:
        """
        Extract verb triplets from the given list of sentences.

        Args:
            text (list): List of sentences.
            entity_types (list): List of entity types to consider for triplets. Default is ["PERSON", "NORP", "FAC", "ORG", "EVENT", "LAW"].

        Returns:
            list: List of verb triplets, where each triplet is a tuple of (subject, relation, object).
        """
        triplets: List[Tuple[str, str, str]] = []
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
