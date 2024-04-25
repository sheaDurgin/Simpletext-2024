import math
from sentence_transformers import util, CrossEncoder, losses
import torch
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CEBinaryClassificationEvaluator, \
    CERerankingEvaluator
from torch.utils.data import DataLoader
from transformers import *
from Preprocessing_tools import *
import numpy as np
from sentence_transformers import InputExample
import sys

seed_value = 42

np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

def fill_dev_samples(dev_samples, qid, query, topic_text, item, label):
    if qid not in dev_samples:
        dev_samples[qid] = {'query': query+" [QSP] "+topic_text, 'positive': set(), 'negative': set()}
    if label == 0:
        label = 'negative'
    else:
        label = 'positive'
    dev_samples[qid][label].add(item[0]+" [TAT] "+item[1])

model_name = sys.argv[1]
model = CrossEncoder(model_name, device='cuda', max_length=512)
tokens = ["[QSP]", "[TAT]"]
model.tokenizer.add_tokens(tokens, special_tokens=True)
model.model.resize_token_embeddings(len(model.tokenizer))


initial_retrieval = read_all_jsons(target_dir="/mnt/netstore1_home/shea.durgin/Baseline_Jsons/")
# 821
# 1271
# initial_retrieval = read_all_jsons(target_dir='/mnt/netstore1_home/shea.durgin/simpletext_qrel_jsons/')
# 836
# 1455

topic_dic = read_topic_file("simpletext_2024_task1_queries.csv")
sample_dic = read_qrel_file("simpletext_2024_task1_train.qrels", initial_retrieval)

n = 1000
if '--val' in sys.argv:
    n = 27

counter = 1
train_samples = []
dev_samples = {}
for qid in sample_dic:
    # stop before testing split
    if int(qid[1:3]) > 15 and '--final' not in sys.argv:
        break
    list_current_sample = sample_dic[qid]
    original_query, topic_text = topic_dic[qid]

    if counter <= n:
        for item in list_current_sample:
            label = item[2]
            if label >= 1:
                label = 1
            train_samples.append(InputExample(texts=[original_query+" [QSP] "+topic_text, item[0]+" [TAT] "+item[1]], label=label))
            if '--val' not in sys.argv:
                fill_dev_samples(dev_samples, qid, original_query, topic_text, item, label)
    else:
        for item in list_current_sample:
            label = item[2]
            fill_dev_samples(dev_samples, qid, original_query, topic_text, item, label)

    counter += 1

print(len(train_samples))

pos_cnt = 0
neg_cnt = 0
for qid in dev_samples:
    pos_cnt += len(dev_samples[qid]['positive'])
    neg_cnt += len(dev_samples[qid]['negative'])

print(pos_cnt)
print(neg_cnt)

num_epochs = 5
model_save_path = f"/mnt/netstore1_home/shea.durgin/{num_epochs}epochs-{model_name.split('/')[-1]}"
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4) # 4
#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CERerankingEvaluator(dev_samples, name='train-eval', write_csv=model_save_path + "/eval_csv")
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    save_best_model=True
)