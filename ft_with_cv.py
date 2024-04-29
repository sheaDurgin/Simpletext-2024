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
from sklearn.model_selection import KFold
import sys
import pandas as pd

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

learning_rates = [2e-05, 1e-05, 9e-06]
epochs = [5, 10]

model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
model_save_path = f"/mnt/netstore1_home/shea.durgin/ft_with_cv-{model_name.split('/')[-1]}"

# initial_retrieval = read_all_jsons(target_dir='/mnt/netstore1_home/shea.durgin/simpletext_qrel_jsons/')
initial_retrieval = read_all_jsons(target_dir="/mnt/netstore1_home/shea.durgin/Baseline_Jsons/")

topic_dic = read_topic_file("simpletext_2024_task1_queries.csv")
qrel_dic = read_qrel_file("simpletext_2024_task1_train.qrels", initial_retrieval)

qids = list(qrel_dic.keys())
kf = KFold(n_splits=5, shuffle=True)

best_lr = None
best_epochs = None
best_avg = None

dic = {}

for num_epochs in epochs:
    dic[num_epochs] = {}
    for lr in learning_rates:
        dic[num_epochs][lr] = 0
        all_scores = []
        for train_index, dev_index in kf.split(qids):
            train_samples = []
            dev_samples = {}
            train_qids = [qids[i] for i in train_index]
            dev_qids = [qids[i] for i in dev_index]

            for qid in train_qids:
                list_current_sample = qrel_dic[qid]
                original_query, topic_text = topic_dic[qid]

                for item in list_current_sample:
                    label = item[2]
                    if label >= 1:
                        label = 1
                    train_samples.append(InputExample(texts=[original_query+" [QSP] "+topic_text, item[0]+" [TAT] "+item[1]], label=label))

            for qid in dev_qids:
                list_current_sample = qrel_dic[qid]
                original_query, topic_text = topic_dic[qid]

                for item in list_current_sample:
                    label = item[2]
                    if label >= 1:
                        label = 1
                    fill_dev_samples(dev_samples, qid, original_query, topic_text, item, label)

            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4) # 4
            evaluator = CERerankingEvaluator(dev_samples, name='train-eval', write_csv=model_save_path + "/eval_csv")
            warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

            model = CrossEncoder(model_name, device='cuda', max_length=512)
            tokens = ["[QSP]", "[TAT]"]
            model.tokenizer.add_tokens(tokens, special_tokens=True)
            model.model.resize_token_embeddings(len(model.tokenizer))

            model.fit(
                train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=num_epochs,
                optimizer_params={'lr': lr},
                warmup_steps=warmup_steps,
                output_path=model_save_path,
                save_best_model=True
            )

            csv_path = model_save_path + "/CERerankingEvaluator_train-eval_results_@10.csv"
            df = pd.read_csv(csv_path)
            all_scores.append(df['NDCG@10'].tail(num_epochs).max())

        print(all_scores)
        avg = sum(all_scores) / len(all_scores)
        print(f"avg: {avg}, lr: {lr}, epochs: {num_epochs}")
        if not best_avg or avg > best_avg:
            best_avg = avg
            best_epochs = num_epochs
            best_lr = lr

        dic[num_epochs][lr] = avg

for num_epochs in dic:
    for lr in dic[num_epochs]:
        print(f"avg: {dic[num_epochs][lr]}, lr: {lr}, epochs: {num_epochs}")

print(f"best_avg: {best_avg}, best_lr: {best_lr}, best_epochs: {best_epochs}")


