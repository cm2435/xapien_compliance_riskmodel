# Risk Engine

The Risk Engine is a Python-based project that models the risk associated with news articles related to a specific company. It utilizes various models and techniques to analyze the temporal patterns, topics, and named entity relationships within the news data.

## Features

- **Temporal Modeling**: The Risk Engine employs a Temporal Model to discover news bursts and identify temporal patterns in the company's news articles.
- **Topic Modeling**: It utilizes a Topic Model to perform general topic modeling on the news articles and extract key semantic topics.
- **Named Entity Recognition (NER) Graph**: The Risk Engine extracts verb triplets involving specific named entity types using a NER Graph.
- **GPT-based Summarization**: It leverages the GPT-3 language model for topic summarization and headline generation.
- **Visualization**: The Risk Engine provides visualizations of the NER Graph and temporal patterns using Matplotlib and NetworkX.

## Installation

1. Clone the repository: `git clone https://github.com/your-username/risk-engine.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Import the necessary classes and functions from the `risk_engine` package.
2. Create an instance of the `RiskEngineBase` class.
3. Load the company data into a dictionary.
4. Call the `model_risk` method on the `RiskEngineBase` instance, passing the company data and optional parameters to enable/disable specific models.
5. Retrieve the risk model output, which includes news bursts, NER graph, and key semantic topics.
6. Visualize the NER graph and temporal patterns using the provided methods.

## Frontend 
To call the frontend, simply run 
```
streamlit run frontend.py
```
If the use_gpt flag is set to be true in the config file config.yaml, an openai api key must be passed. 

## Example usage 

```python
from risk_engine import RiskEngineBase

# Create an instance of the RiskEngineBase class
risk_engine = RiskEngineBase()

# Load the company data into a dictionary
company_data = {
    "SearchResults": [
        {
            "date": "2023-06-01",
            "title": "Article 1",
            "snippet": "Snippet 1"
        },
        {
            "date": "2023-06-02",
            "title": "Article 2",
            "snippet": "Snippet 2"
        },
        # Add more articles...
    ]
}

# Model the risk using the RiskEngine
output = risk_engine.model_risk(
    data=company_data,
    temporal_model=True,
    topic_model=True,
    ner_graph=True,
    use_gpt=False
)

# Access the risk model output
news_bursts = output["news_bursts"]
ner_graph = output["ner_graph"]
topics = output["topics"]

# Visualize the NER graph
risk_engine.visualise_graph(ner_graph)

# Visualize the temporal patterns
risk_engine.plot_dates(company_data)
