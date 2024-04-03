import matplotlib.pyplot as plt

qrel_dic = {}
with open('simpletext_2024_task1_train.qrels', 'r') as f:
    for line in f.readlines():
        topic_id, _, doc_id, label = line.strip().split()
        if topic_id not in qrel_dic:
            qrel_dic[topic_id] = {}
        qrel_dic[topic_id][doc_id] = int(label)

result_dic = {}
with open('final_results.txt', 'r') as f:
    rank = 1
    prev_topic_id = None
    for line in f.readlines():
        topic_id, _, doc_id, _, score, _ = line.strip().split()
        if topic_id != prev_topic_id:
            prev_topic_id = topic_id
            rank = 1
        if topic_id not in qrel_dic or doc_id not in qrel_dic[topic_id]:
            continue
        if topic_id not in result_dic:
            result_dic[topic_id] = {}
        result_dic[topic_id][doc_id] = rank
        rank += 1

mrr_per_topic = {}
for topic_id, result_topic_dic in result_dic.items():
    if topic_id not in qrel_dic:
        continue
    qrel_topic_dic = qrel_dic[topic_id]
    sorted_results = sorted(result_topic_dic.items(), key=lambda x: x[1])
    for doc_id, rank in sorted_results:
        label = qrel_topic_dic[doc_id]
        if label > 0:
            mrr_per_topic[topic_id] = round(1 / rank, 2)
            break

sorted_mrr_per_topic = sorted(mrr_per_topic.items(), key=lambda x: x[1], reverse=True)
topic_ids, mrr_values = zip(*sorted_mrr_per_topic)

print(f"Average MRR: {sum(mrr_values) / len(mrr_values)}")

for topic_id, mrr in zip(topic_ids, mrr_values):
    print(f"{topic_id} : {mrr}")

plt.figure(figsize=(16, 8))
plt.bar(topic_ids, mrr_values)
plt.xlabel('Topic ID')
plt.ylabel('MRR')
plt.grid(False)
plt.show()
plt.xticks(rotation=60)
plt.savefig('mrr_per_topic.png')