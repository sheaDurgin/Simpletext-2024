# SimpleText 2024 Task 1

This repository includes all of the codes for my submission to SimpleText Task 1.

## Table of Contents

- [Installation](#Installation)
- [Text Files](#Text-Files)
- [Steps to Run](#Steps-to-Run)
- [Model Details](#Model-Details)
- [Results](#Final-Results-on-Unseen-2023-Test-Set)

## Installation

To run this code, you will need access to the [dataset](https://simpletext-project.com/2024/en/) for task 1 of SimpleText CLEF lab, and login details to access their servers. I will be assuming you have access for the rest of the explanation.

First create a virtual environment, for example

    python -m venv myenv
    source myenv/bin/activate

Install libraries using pip and the requirements.txt

    pip install -r requirements.txt
    
Clone the repository all to one folder to properly run. Directories may need to be changed to fit your machine.

## Text Files
- baseline.txt -> the top 95 results from ElasticSearch in a three stage search (`"{query}"`, `{query}`, `{topictext}`)
- selective.txt -> the top results from ElasticSearch in a one stage search (`"{query}"`)
- rr_baseline.txt -> results from cross_encoder.py using baseline.txt as argument
- final_results.txt -> results from combine_scores.py using rr_baseline.txt and selective.txt

## Steps to Run

- Get the SimpleText dataset from CLEF and have both the qrels and topics csv file in repository
- Update the config.json file with user, password, and ElasticSearch URL, to log in to Elastic Search
- run `python run_everything.py` (can modify names of files and directories in this file)

## Model Details

The final results come from a combination of a re-ranked baseline retrieval using our finetuned ms-marco-MiniLM-L-6-v2 cross encoder and the selective retrieval from ElasticSearch. The cross encoder does its own re-ranking of the top 95 results from ElasticSearch, then that output is directed into a comination program that does a final re-ranking using a combination of the cross encoder ranking and the selective retrieval. This is all based on the assumption that when ElasticSearch and the cross-encoder rank a document highly, there is a higher chance that it is more relevant than a document rated highly on only one system.

## Final Results on Unseen 2023 Test Set

    MRR: 0.8946078431372548
    NDCG@10: 0.5957062060467501
    MAP: 0.2801930878426999
