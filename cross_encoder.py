from Preprocessing_tools import *
import json
import os
import glob
from sentence_transformers import CrossEncoder
import torch
import sys
from tqdm import tqdm

# get abstracts from jsons
def json_abstracts(doc_ids, topic, base_dir):
    json_files = glob.glob(f"{base_dir}*{topic}*")
    if not json_files:
        raise ValueError(f"No JSON file found for topic '{topic}'")

    json_path = json_files[0]
    with open(json_path) as f:
        data = json.load(f)
        abstracts = {}
        for doc_id in doc_ids:
            abstract = next((hit['_source']['abstract'] for hit in data['hits']['hits'] if hit['_id'] == doc_id), None)
            title = next((hit['_source']['title'] for hit in data['hits']['hits'] if hit['_id'] == doc_id), None)
            abstracts[doc_id] = title + " [TAT] " + abstract
        return abstracts

# rerank results with cross encoder
def cross_encoder_ranking(topic_dic, trec_file_path, model, base_dir):
    initial_retrieval = read_trec_file(trec_file_path)
    cross_encoder_ranked_docs = []

    for topic in tqdm(topic_dic, desc="Processing topics"):
        query, topic_text = topic_dic[topic]
        corrected_topic = topic.replace("_", ".")
        
        if corrected_topic not in initial_retrieval:
            continue
        
        initial_ret_topic = initial_retrieval[corrected_topic]
        doc_ids = [doc_id for doc_id in initial_ret_topic]
        abstract_dict = json_abstracts(doc_ids, topic, base_dir)
        
        corpus = [abstract_dict[doc_id] for doc_id in initial_ret_topic]
        ce_scores = [model.predict([[query + " [QSP] " + topic_text, abstract]])[0] for abstract in corpus]
        k_value = min(100, len(ce_scores))
        top_results = torch.topk(torch.Tensor(ce_scores), k=k_value)

        ranked_docs = [(initial_ret_topic[int(idx)], score.item()) for score, idx in zip(top_results[0], top_results[1])]
        cross_encoder_ranked_docs.append((topic.replace("_", "."), ranked_docs))

    # Normalize scores for each topic by the corresponding max score using the sigmoid function
    for i, (topic_id, ranked_docs) in enumerate(cross_encoder_ranked_docs):
        max_score = max(score for _, score in ranked_docs)
        ranked_docs_norm = [(doc_id, torch.sigmoid(torch.tensor(score)) / torch.sigmoid(torch.tensor(max_score)).item()) for doc_id, score in ranked_docs]
        cross_encoder_ranked_docs[i] = (topic_id, ranked_docs_norm)

    return cross_encoder_ranked_docs


# transform trec file to dictionary
def read_trec_file(trec_file_path):
    result_dict = {}
    with open(trec_file_path, 'r') as f:
        for line in f.readlines():
            topic_id, _, doc_id, rank, score, _ = line.strip().split()
            if topic_id not in result_dict:
                result_dict[topic_id] = []
            result_dict[topic_id].append(doc_id)
    return result_dict

# write the results of cross encoder in trec format
def write_rankings(rankings, file_path):
    with open(file_path, "w") as f:
        for query_id, docs in rankings:
            for rank, (doc_id, score) in enumerate(docs, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} Top_Gap\n")

def main():
    args = sys.argv[1:]
    # file that holds elastic search results
    read_file = args[0]
    # new file to print results to
    results_file = args[1]
        
    print("Reading json")
    base_dir = args[2]
    initial_retrieval = read_all_jsons(target_dir=base_dir)
    topic_dic = read_topic_file("simpletext_2024_task1_queries.csv")

    model_path = args[3]
    model = CrossEncoder(model_path)

    print("CrossEncoder ranking")
    cross_encoder_ranked_docs = cross_encoder_ranking(topic_dic, read_file, model, base_dir)

    print("Writing CrossEncoder ranked file")
    write_rankings(cross_encoder_ranked_docs, results_file)

if __name__ == "__main__":
    main()
