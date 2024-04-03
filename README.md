# SimpleText 2024 Task 1

This repository includes all of the codes for my submission to SimpleText Task 1.

## Table of Contents

- [Installation](#Installation)
- [Text Files](#Text-Files)
- [Steps to Run](#Steps-to-Run)
- [Model Details](#Model-Details)
- [Results](#Top-100-Per-Query-Baseline-Results)
- [Conclusion](#Conclusion)

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
