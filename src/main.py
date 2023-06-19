import numpy as np
import pandas as pd
import json

from modelling.temporal import TemporalModel
from modelling.topic_models import TopicModel

class RiskEngineBase:
    def __init__(self):
        self.temporal_model = TemporalModel()
        self.topic_model = TopicModel()

    @staticmethod
    def _parse_company_data(data: dict):
        """
        Parse the company data dictionary and return a DataFrame.

        Args:
            data (dict): Company data dictionary.

        Returns:
            pd.DataFrame: Parsed DataFrame.
        """
        df = pd.DataFrame(data["SearchResults"])
        df.columns = df.columns.str.lower()
        df["full_text"] = df.title + df.snippet
        df = df.drop_duplicates(subset="full_text")
        return df

    def model_risk(
        self,
        data: dict,
        temporal_model: bool = True,
        visualise: bool = True,
        topic_model: bool = True,
        **kwargs
    ) -> dict:
        """
        Model the risk using temporal and topic models.

        Args:
            data (dict): Company data dictionary.
            temporal_model (bool): Flag to enable temporal modeling. Default is True.
            visualise (bool): Flag to enable visualization. Default is True.
            topic_model (bool): Flag to enable topic modeling. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Risk model output schema.
        """
        # Load data
        df = self._parse_company_data(data=data)

        # Discover news bursts and model them
        if temporal_model:
            df["temporal_label"] = self.temporal_model.predict(df["date"])
        if visualise:
            pass

        # Do general topic modelling and find key docs for each
        if topic_model:
            unique_titles = df.title.drop_duplicates().values
            topic_df = self.topic_model.get_topics(topic_text=unique_titles)
            df = pd.merge(
                df, topic_df, left_on=["title"], right_on=["document"], how="inner"
            )
            # Drop the redundant 'document' column from the merged DataFrame
            df = df.drop("document", axis=1)

        # Parse out all of the 'bursts' of news and their date ranges
        output_schema = {}
        output_schema["news_bursts"] = self.temporal_model.fitted_intervals

        # Postprocess the topic model to get the key semantic topics
        output_schema["topics"] = []
        for topic in df.topic.unique():
            if topic != -1:  # Filter out the 'non-topic' label
                filtered_df = df[df.topic == topic].head(1)

                topic_dict = {
                    "theme": filtered_df["representation"].values[0],
                    "top_titles": filtered_df["representative_docs"].tolist(),
                    "extracted_keywords": filtered_df["top_n_words"].tolist(),
                }
                topic_dict["top_snippets"] = self.topic_model.find_duplicates(
                    titles=filtered_df["snippet"].tolist(),
                    title_docs=filtered_df["representative_docs"].tolist()[0],
                )
                output_schema["topics"].append(topic_dict)

        # Rank articles for each topic found
        print(output_schema)




if __name__ == "__main__":
    x = RiskEngineBase()
    with open("/home/cm2435/Desktop/xapien_task/data/NiramaxTextData.json", "r") as f:
        file = json.load(f)

    print(x.model_risk(file))
