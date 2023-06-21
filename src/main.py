import json
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from sklearn.cluster import DBSCAN

from .modelling.llm_wrapper import ChatGPTWrapper
from .modelling.ner_graph import NerNetworkModel
from .modelling.temporal import TemporalModel
from .modelling.topic_models import TopicModel


class RiskEngineBase:
    def __init__(self):
        self.temporal_model = TemporalModel()
        self.topic_model = TopicModel()
        self.llm_wrapper = ChatGPTWrapper()
        self.ner_model = NerNetworkModel()

    @staticmethod
    def _parse_company_data(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data["SearchResults"])
        df.columns = df.columns.str.lower()
        df["full_text"] = df.title + df.snippet
        df = df.drop_duplicates(subset="full_text")
        return df

    def model_risk(self, data: dict, **kwargs) -> dict:
        df = self._parse_company_data(data=data)
        output_schema = {}

        if kwargs.get("temporal_model", True):
            df["temporal_label"] = self.temporal_model.predict(df["date"])
            output_schema["news_bursts"] = self.temporal_model.fitted_intervals

        if kwargs.get("topic_model", True):
            unique_titles = df.title.drop_duplicates().values
            topic_df = self.topic_model.get_topics(topic_text=unique_titles)
            df = pd.merge(
                df, topic_df, left_on=["title"], right_on=["document"], how="inner"
            )
            df = df.drop("document", axis=1)

        if kwargs.get("ner_graph", True):
            output_schema["ner_graph"] = self.ner_model.extract_verb_triplets(
                text=df["full_text"],
                entity_types=["PERSON", "NORP", "FAC", "ORG", "EVENT", "LAW"],
            )

        output_schema["topics"] = []
        for topic in df.topic.unique():
            if topic != -1:
                filtered_df = df[df.topic == topic]
                topic_dict = {
                    "theme": filtered_df["representation"].values[0],
                    "top_titles": filtered_df["representative_docs"].tolist()[0],
                    "extracted_keywords": filtered_df["top_n_words"].tolist()[0],
                }

                if kwargs.get("use_gpt", True):
                    gpt_description = self.llm_wrapper.predict(
                        titles=filtered_df["representative_docs"].tolist(),
                        articles=None,
                        task="summary",
                    )
                    if gpt_description != "<FAILED>":
                        topic_dict["theme"] = gpt_description

                topic_dict["top_snippets"] = self.topic_model.find_duplicates(
                    titles=filtered_df["representative_docs"].tolist()[0],
                    title_docs=filtered_df["snippet"].tolist(),
                )
                output_schema["topics"].append(topic_dict)

        return output_schema

    def visualise_graph(self, triplets: List[List[str]]):
        graph = nx.DiGraph()
        for subject, relation, object_ in triplets:
            graph.add_edge(subject, object_, label=relation)

        plt.figure(figsize=(18, 12))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")
        nx.draw_networkx_edges(graph, pos, arrows=True, edge_color="gray")
        nx.draw_networkx_labels(graph, pos, font_size=12, font_color="black")
        edge_labels = nx.get_edge_attributes(graph, "label")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        plt.title("Dependency Graph")
        plt.axis("off")
        plt.show()

    def plot_dates(self, data: dict):
        df = self._parse_company_data(data)
        dates = df["date"].tolist()
        converted_dates = []

        for date in dates:
            if date is not None:
                date_str = f"{date['Year']}-{date['Month']}-{date['Day']}"
                converted_date = datetime.strptime(date_str, "%Y-%m-%d")
                converted_dates.append(converted_date)

        clustering = DBSCAN(eps=365 / 2, min_samples=5).fit(
            date2num(converted_dates).reshape(-1, 1)
        )
        date_clusters = []
        for i, label in enumerate(clustering.labels_):
            date_clusters.append((converted_dates[i], label))

        df = pd.DataFrame(date_clusters, columns=["date", "label_temporal"])

        plt.figure(figsize=(8, 6))
        for label in df["label_temporal"].unique():
            cluster_dates = df[df["label_temporal"] == label]["date"]
            cluster_converted_dates = [
                datetime.strptime(f"{date.strftime('%Y-%b-%d')}", "%Y-%b-%d")
                for date in cluster_dates
                if date is not None
            ]
            plt.scatter(
                cluster_converted_dates,
                range(len(cluster_converted_dates)),
                marker="o",
                label=f"Cluster {label}",
            )

        plt.xlabel("Date")
        plt.ylabel("")
        plt.title("Dates on a Number Line")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import openai

    risk_engine = RiskEngineBase()
    with open(
        "/Users/charliemasters/Desktop/xapien_compliance_riskmodel/data/NiramaxTextData.json",
        "r",
    ) as f:
        file = json.load(f)

    output = risk_engine.model_risk(
        data=file, temporal_model=True, topic_model=True, ner_graph=True, use_gpt=False
    )
    print(output)
