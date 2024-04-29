from collections import defaultdict
from scipy.stats import wilcoxon
import numpy as np
import sys

# File paths
new_results = sys.argv[1]
old_results = sys.argv[2]
qrels = sys.argv[3]

# Read data from files
with open(new_results, 'r') as f:
    file1_lines = f.readlines()

with open(old_results, 'r') as f:
    file2_lines = f.readlines()

# Parse relevance judgments
qrel_dic = defaultdict(dict)
with open(qrels, 'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        query_id, doc_id, relevance = parts[0], parts[2], int(parts[3])
        qrel_dic[query_id][doc_id] = relevance

# Parse data from ranked lists
def parse_data(lines):
    parsed_data = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        parsed_data[parts[0]].append(parts[2])  # query_id to ranked list of document_id, 1st to N
    return parsed_data

file1_data = parse_data(file1_lines)
file2_data = parse_data(file2_lines)

# Calculate Discounted Cumulative Gain (DCG) at k
def dcg_at_k(rank_list, relevant_documents, k):
    dcg = 0.0
    for i in range(min(k, len(rank_list))):
        doc_id = rank_list[i]
        relevance = relevant_documents.get(doc_id, 0)  # Get relevance level (default to 0 if not found)
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    return dcg

# Calculate Normalized Discounted Cumulative Gain (NDCG) at k
def ndcg_at_k(rank_list, relevant_documents, k):
    ideal_rank_list = sorted(relevant_documents, key=lambda x: relevant_documents[x], reverse=True)
    ideal_dcg = dcg_at_k(ideal_rank_list, relevant_documents, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(rank_list, relevant_documents, k) / ideal_dcg

# Calculate the metric (NDCG) for each query
def calculate_metric(file_data, k=10):
    metrics = []
    for query_id, ranked_list in file_data.items():
        # Only 2023 unseen test set
        if len(query_id) < 6 and ((query_id[0] == 'G' and int(query_id[1:3]) > 15) or (query_id[0] == 'T' and int(query_id[1:3]) < 6)):
            relevant_documents = qrel_dic[query_id]
            metric_value = ndcg_at_k(ranked_list, relevant_documents, k=k)
            metrics.append(metric_value)
    return metrics

# Calculate metrics for both files
file1_metrics = calculate_metric(file1_data)
file2_metrics = calculate_metric(file2_data)

print(len(file1_metrics))
print(len(file2_metrics))

# Perform Wilcoxon signed-rank test
statistic, p_value = wilcoxon(file1_metrics, file2_metrics)

# Output results
print("Wilcoxon signed-rank test:")
print("Statistic:", statistic)
print("P-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("There is a significant difference between the ranked lists.")
else:
    print("There is no significant difference between the ranked lists.")
