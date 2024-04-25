# SimpleText 2024 Task 1

This repository includes all of the codes for my submission to SimpleText Task 1.

## Table of Contents

- [Installation](#Installation)
- [Text Files](#Text-Files)
- [Steps to Run](#Steps-to-Run)
- [Model Details](#Model-Details)
- [Results](#Top-100-Per-Query-Baseline-Results)

## Installation

To run this code, you will need access to the [dataset](https://simpletext-project.com/2024/en/) for task 1 of SimpleText CLEF lab, and login details to access their servers. I will be assuming you have access for the rest of the explanation.

The necessary installs for the code are as such

    torch
    tqdm
    sentence_transformers
    transformers
    textstat
    ranx
    markdown
    matplotlib

You can install them using pip:

    pip install torch tqdm sentence_transformers transformers textstat ranx markdown matplotlib
    
Clone the repository all to one folder to properly run. Directories may need to be changed to fit your machine.

## Text Files
- baseline.txt -> the top 100 results from elastic search for each query
- selective.txt -> the top results (up to 100) from elastic search for each query (no work done to get extra results)
- rr_baseline.txt -> results from cross_encoder.py using baseline.txt as argument
- final_results.txt -> results from combine_scores.py using rr_baseline.txt and selective.txt

## Steps to Run

- Get the SimpleText dataset from CLEF and have both the qrels and topics csv file in repository
- Create a config.json file with user, password, and ElasticSearch URL Prefix, to log in to Elastic Search
- run run_everything.py (can modify names of files and directories in this file)

## Model Details

The final results come from a combination of the ms-marco-electra-base cross encoder and the two baseline results from elastic search. The cross encoder does its own reranking of the top 100 results from elastic search, then that output is directed into a comination program that does a final reranking using a combination of the cross encoder ranking and the selective baseline results. This results in higher NDCG and MAP scores than any of the individual results.

## Baseline Results

    MRR: 0.6822189606436181
    NDCG@10: 0.374
    MAP: 0.438
    flesch average: 28.673
    smog average: 14.792
    Coleman Liau average: 15.923
    
## Selective Results (Gives less total results)

    MRR: 0.18872549019607843
    NDCG@10: 0.409
    MAP: 0.456
    flesch average: 28.673
    smog average: 14.792
    Coleman Liau average: 15.923
    
## Reranking Results

    MRR: 0.7033029878618113
    NDCG@10: 0.307
    MAP: 0.316
    flesch average: 28.673
    smog average: 14.792
    Coleman Liau average: 15.923

## Final Results

    MRR: 0.8105742296918768
    NDCG@10: 0.460
    MAP: 0.506
    flesch average: 28.673
    smog average: 14.792
    Coleman Liau average: 15.923
