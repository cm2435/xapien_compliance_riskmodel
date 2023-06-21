import json
from typing import List

import openai
import pandas as pd
import streamlit as st

from src.main import RiskEngineBase

def visualize_topics(topics):
    for topic in topics:
        st.subheader(topic["theme"])
        st.write("Top Titles:")
        for title in topic["top_titles"]:
            st.write("- " + title)
        st.write("Extracted Keywords:")
        st.write("- " + topic["extracted_keywords"])
        st.write("Top Snippets:")
        for snippet, score in topic["top_snippets"]:
            st.write("- Snippet:", snippet)
            st.write("  Score:", score)
            st.write("")  # Add an empty line for spacing



# Define the Streamlit app
def main():
    st.title("Risk Engine Dashboard")

    # Upload and process data
    uploaded_file = st.file_uploader("Upload Company Data", type=["json"])
    if uploaded_file is not None:
        data = json.load(uploaded_file)

        # Run the risk model
        risk_model_output = risk_engine.model_risk(
            data=data,
            temporal_model=True,
            topic_model=True,
            ner_graph=True,
            use_gpt=True
        )

        # Display the output JSON
        st.subheader("Risk Model Output")
        st.json(risk_model_output)

        visualize_topics(risk_model_output['topics'])
        # Visualize the NER Graph
        st.subheader("NER Graph Visualization")
        ner_graph = risk_model_output.get("ner_graph")
        if ner_graph is not None:
            ner_graph_plot = risk_engine.visualise_graph(ner_graph)
            st.pyplot(ner_graph_plot)

        # Visualize the Temporal Clustering
        st.subheader("Temporal Clustering Plot")
        temporal_clustering_plot = risk_engine.plot_dates(data)
        st.pyplot(temporal_clustering_plot)

# Run the Streamlit app
if __name__ == "__main__":
    # Instantiate the RiskEngineBase class
    risk_engine = RiskEngineBase()

    openai.api_key = "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"
    main()
