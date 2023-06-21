import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.modelling.llm_wrapper import ChatGPTWrapper


@pytest.fixture
def chatgpt_wrapper():
    return ChatGPTWrapper()


def test_predict_returns_response(chatgpt_wrapper):
    titles = ["title 1", "title 2"]
    articles = ["article 1", "article 2"]
    task = "summary"

    response = chatgpt_wrapper.predict(titles, articles, task)

    assert response is not None


def test_parse_prompt_for_summary_task(chatgpt_wrapper):
    titles = ["title 1", "title 2"]
    bodies = ["body 1", "body 2"]
    task = "summary"

    prompt = chatgpt_wrapper._parse_prompt(titles, bodies, task)

    expected_prompt = (
        "You are a helpful expert assistant being asked to analyse the risk of news for a client."
        " The news data will be text based and focus on a specific company. The goal is to help a layperson"
        " be able to understand what a company may be involved with and when. It is vital to catch risky dealings"
        " of companies we analyse.\n\nIf the company is mentioned in an article, it does not mean it is necessarily"
        " risky. For example, a fraud-prosecuting law firm is not risky, a company being prosecuted for fraud is risky."
        "\n\n\n title 1\n title 2"
    )

    assert prompt == expected_prompt
