from typing import List

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TopicModel:
    """
    A class for topic modeling text data using BERTopic.

    Attributes:
        embedding_model (SentenceTransformer): SentenceTransformer model for text embeddings.
        topic_model (BERTopic): BERTopic model for performing topic modeling.
        _is_fitted (bool): Flag indicating whether the topic model has been fitted.
    """

    def __init__(self):
        self.embedding_model: SentenceTransformer = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.topic_model: BERTopic = BERTopic(embedding_model=self.embedding_model)
        self._is_fitted: bool = False

    def fit(self, topic_text: List[str]) -> None:
        """
        Fit a topic model to a List of text data.

        Args:
            topic_text (list): List of topic text.

        Returns:
            None
        """
        self._is_fitted = True
        self.topic_model.fit(topic_text)

    def get_topics(self, topic_text: List[str]) -> pd.DataFrame:
        """
        Get a series of topics and useful metadata about a series of text objects

        Args:
            topic_text (list): List of topic text.

        Returns:
            DataFrame: DataFrame with topic information.
        """
        if not self._is_fitted:
            self.fit(topic_text)

        topic_df: pd.DataFrame = self.topic_model.get_document_info(topic_text)
        representative_docs = self.topic_model.get_representative_docs()
        topic_df.columns = topic_df.columns.str.lower()
        topic_df["representative_docs"] = topic_df["topic"].map(representative_docs)

        return topic_df

    def find_duplicates(
        self, titles: List[str], title_docs: List[str], max_return: int = 5
    ) -> List[str]:
        """
        Find the most relevent snippets to the indicative headlines extracted for a topic by ranking
        them based on semantic similarity.

        Args:
            titles (list): List of the top 3 titles for a topic.
            title_docs (list): List of article snippets to be ranked.
            max_return (int): Maximum number of duplicates to return. Default is 5.

        Returns:
            list: Sorted list of duplicate titles with similarity scores.
        """
        title_embedding = self.embedding_model.encode(titles)
        embeddings = self.embedding_model.encode(title_docs)
        similarity_matrix = cosine_similarity(
            embeddings, np.average(title_embedding, axis=0).reshape(1, -1)
        )

        topic_list: List[str] = []
        for i, title in enumerate(title_docs):
            # Do not append duplicates
            duplicate = False
            for tuple_item in topic_list:
                if title == tuple_item[0]:
                    duplicate = True
            if not duplicate:
                topic_list.append((title, similarity_matrix[i][0]))

        return sorted(topic_list, key=lambda x: x[1], reverse=True)[:max_return]
