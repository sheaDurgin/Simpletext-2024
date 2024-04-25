import csv
import json
import os

def read_json(filename: str):
    # reads one json file
    with open(filename) as f_in:
        return json.load(f_in)

def read_all_jsons(target_dir):
    # takes in teh directory of topics and return dictionary of topics with id corresponding to qrel files
    dict_top_res = {}
    for file in os.listdir(target_dir):
        temp_dic = read_json(target_dir+file)
        hits = temp_dic['hits']['hits']
        temp_dic_result = {}
        for hit in hits:
            source = hit['_source']
            paper_id = source['id']
            title = source['title']
            abstract = source['abstract']
            temp_dic_result[paper_id] = (title, abstract)

        query_id = file.split(".")[0]
        # if len(temp_dic_result) < 2000:
        #     print(query_id + "\t" + str(len(temp_dic_result)))
        dict_top_res[query_id] = temp_dic_result

    return dict_top_res

def read_one_json(target_dir):
    # takes in teh directory of topics and return dictionary of topics with id corresponding to qrel files
    dict_top_res = {}
    temp_dic = read_json(target_dir)
    hits = temp_dic['hits']['hits']
    temp_dic_result = {}
    for hit in hits:
        source = hit['_source']
        paper_id = source['id']
        title = source['title']
        abstract = source['abstract']
        dict_top_res[paper_id] = (title, abstract)

    return dict_top_res


def read_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab; to be passed to BERT/PyTerrier methods
    result = {}
    with open(topic_filepath, "r") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        pre_qid = ""
        counter = 1
        for line in reader:
            original_query = line[-1]
            topic_text = line[1]
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result[q_id + "_" + str(counter)] = (original_query, topic_text)
    return result

def read_qrel_file(qrel_path, initial_retrieval, tsv_bool=False):
    dic = {}
    with open(qrel_path, 'r') as f:
        if tsv_bool:
            reader = csv.reader(f, delimiter='\t')
        else:
            reader = csv.reader(f, delimiter=' ')
        for line in reader:
            qid, _, paper_id, label = line
            qid = qid.replace('.', '_')
            paper_id = int(paper_id)
            if paper_id not in initial_retrieval[qid]:
                continue
            title, abstract = initial_retrieval[qid][paper_id]
            if qid not in dic:
                dic[qid] = []
            dic[qid].append((title, abstract, int(label)))

    return dic    

if __name__ == '__main__':
    initial_retrieval = read_all_jsons(target_dir="Baseline_Jsons/")
    dic = read_qrel_file('simpletext_2023_task1_train.qrels', initial_retrieval)