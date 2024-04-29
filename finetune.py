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
import os
import shutil

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

prefix = sys.argv[4]
initial_retrieval = read_all_jsons(target_dir=f"{prefix}Baseline_Jsons/")

topic_dic = read_topic_file("simpletext_2024_task1_queries.csv")
sample_dic = read_qrel_file("simpletext_2024_task1_train.qrels", initial_retrieval)

train_samples = []
dev_samples = {}
for qid in sample_dic:
    # break if fine-tuning for 2023 test set
    if qid[0] == "G" and int(qid[1:3]) > 15 and '--final' not in sys.argv: 
        break
    list_current_sample = sample_dic[qid]
    original_query, topic_text = topic_dic[qid]

    for item in list_current_sample:
        label = item[2]
        if label >= 1:
            label = 1
        if qid[0:3] == "G10" or qid[0:3] == "G11":
            fill_dev_samples(dev_samples, qid, original_query, topic_text, item, label)
        else:
            train_samples.append(InputExample(texts=[original_query+" [QSP] "+topic_text, item[0]+" [TAT] "+item[1]], label=label))

model_save_path = f"{prefix}final-{model_name.split('/')[-1]}"
if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)

num_epochs = int(sys.argv[2])
lr = float(sys.argv[3])

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4) # 4
evaluator = CERerankingEvaluator(dev_samples, name='train-eval', write_csv=model_save_path + "/eval_csv")
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    optimizer_params={'lr': lr},
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    save_best_model=True
)