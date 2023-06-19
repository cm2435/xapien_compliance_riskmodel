import json
import time
from pathlib import Path
from typing import List, Union

import openai


class ChatGPTWrapper:
    def __init__(self):
        self._assistant_prompt = "You are a helpful expert assistant being asked to analyse the risk of news for a client. The news data will be text based and focus on a specific company. The goal is to help a layperson be able to understand what a company may be involved with and when. It is vital to catch risky dealings of companies we analyse. \n\nIf the company is mentioned in an article, it does not mean it is necessarily risky. For example, a fraud-prosecuting law firm is not risky, a company being prosecuted for fraud is risky. \n\n"
        self.prompt_templates = json.load(
            open(str(Path(__file__).parent / "prompt_store.json"), "rb")
        )
        ''' assert openai.api_key is not None, \
                """
                To use this model, an openAI api key must have been set in the environment previously
                Please verify that this is the case.
                """     '''

    def predict(
        self, titles: Union[List[str], str], articles: Union[List[str], str], task: str
    ):
        prompt = self._parse_prompt(titles=titles, bodies=articles, task=task)
        response = self.invoke_chatgpt(prompt=prompt)
        print(response)
        if "<FAILED>" in response:
            return None

        return response

    def invoke_chatgpt(self, prompt: str, max_retries: int = 3, backoff_time: int = 2):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self._assistant_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )

                if len(response.choices) > 0:
                    return response.choices[0].message.content.strip()

                return None
            except openai.error.RateLimitError:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff

        return None

    def _parse_prompt(
        self, titles: Union[List[str], str], bodies: Union[List[str], str], task: str
    ):
        """ """
        prompt = self.prompt_templates[task]
        if task == "summary":
            # Handle nested lists
            if isinstance(titles[0], list):
                titles = titles[0]
            substring = ""
            for title in titles:
                substring += "\n" + title
            prompt = prompt.format(substring)

        return prompt


if __name__ == "__main__":
    x = ChatGPTWrapper()
    testcase = [
        "man stabbed outside shops ",
        "niramax murder case 2",
        "article title 3",
    ]
    print(x.predict(testcase, 0, task="summary"))
