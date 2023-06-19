from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic


class TopicModel:
    """
    A class for topic modeling using BERTopic.

    Attributes:
        embedding_model (SentenceTransformer): SentenceTransformer model for text embeddings.
        topic_model (BERTopic): BERTopic model for topic modeling.
        _is_fitted (bool): Flag indicating whether the model has been fitted.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_model = BERTopic(embedding_model=self.embedding_model)
        self._is_fitted = False

    def fit(self, topic_text: List[str]):
        """
        Fit the topic model to the given text.

        Args:
            topic_text (list): List of topic text.

        Returns:
            None
        """
        self._is_fitted = True
        self.topic_model.fit(topic_text)

    def get_topics(self, topic_text: List[str]):
        """
        Get the topics for the given text.

        Args:
            topic_text (list): List of topic text.

        Returns:
            DataFrame: DataFrame with topic information.
        """
        if not self._is_fitted:
            self.fit(topic_text)

        topic_df = self.topic_model.get_document_info(topic_text)
        topic_df.columns = topic_df.columns.str.lower()

        return topic_df

    def find_duplicates(self, titles: List[str], title_docs: List[str], max_return: int = 5):
        """
        Find duplicates based on title similarity.

        Args:
            titles (list): List of titles.
            title_docs (list): List of title documents.
            max_return (int): Maximum number of duplicates to return. Default is 5.

        Returns:
            list: Sorted list of duplicate titles with similarity scores.
        """
        embeddings = self.embedding_model.encode(titles)
        title_embedding = self.embedding_model.encode(title_docs)
        similarity_matrix = cosine_similarity(embeddings, np.average(title_embedding, axis=0).reshape(1, -1))

        topic_list = []
        for i, title in enumerate(titles):
            topic_list.append((title, similarity_matrix[i][0]))

        return sorted(topic_list, key=lambda x: x[1])[:max_return]
