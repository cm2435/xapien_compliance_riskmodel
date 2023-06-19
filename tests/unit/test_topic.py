import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.modelling.topic_models import TopicModel


@pytest.fixture
def topic_model():
    return TopicModel()


def test_fit_sets_is_fitted_true(topic_model):
    topic_text = ["topic 1", "topic 2", "topic 3"]

    topic_model.fit(topic_text)

    assert topic_model._is_fitted is True


def test_get_topics_returns_dataframe(topic_model):
    topic_text = ["topic 1", "topic 2", "topic 3"]

    result = topic_model.get_topics(topic_text)

    assert isinstance(result, pd.DataFrame)


def test_find_duplicates_returns_sorted_list(topic_model):
    titles = ["title 1", "title 2", "title 3"]
    title_docs = ["document 1", "document 2", "document 3"]

    result = topic_model.find_duplicates(titles, title_docs)

    assert isinstance(result, list)
    assert len(result) <= 5
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    assert sorted(result, key=lambda x: x[1], reverse=True) == result
