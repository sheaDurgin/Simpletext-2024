import requests
import csv
import json
import os
from tqdm import tqdm

def get_extra_results(obj, query_to_es, line, auth):
    # remove quote
    hits_count = len(obj['hits']['hits'])
    if hits_count<2000:
        if query_to_es.endswith("\""):
            original_query = query_to_es.split("q=")[1][1:-1]
            remaining = 2000-hits_count
            remade_query = query_to_es.split("q=")[0] + "q=" + original_query + "&size="+str(remaining)
            result = requests.get(remade_query, auth=auth).content.decode("utf-8")
            obj2 = json.loads(result)
            for item in obj2['hits']['hits']:
                if item not in obj['hits']['hits']:
                    obj['hits']['hits'].append(item)

    # replace with topic_text
    hits_count = len(obj['hits']['hits'])
    if hits_count < 2000:
            original_query = query_to_es.split("q=")[1][1:-1]
            remaining = 2000 - hits_count
            remade_query = query_to_es.split("q=")[0] + "q=" + line[1] + "&size=" + str(remaining)
            result = requests.get(remade_query, auth=auth).content.decode("utf-8")
            obj2 = json.loads(result)
            for item in obj2['hits']['hits']:
                if item not in obj['hits']['hits']:
                    obj['hits']['hits'].append(item)

def download(target_dir, num_of_results, extra_results, auth):
    with open("SP12023topics.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        reader_len = sum(1 for line in reader) - 1
        f.seek(0)
        next(reader)
        pre_qid = ""
        counter = 1
    
        for line in tqdm(reader, desc='Creating JSONS', total=reader_len):
            query_to_es = line[-1]
            url = query_to_es+ "&size=" + str(num_of_results)
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result = requests.get(url, auth=auth).content.decode("utf-8")
            obj = json.loads(result)
            
            if extra_results:
                get_extra_results(obj, query_to_es, line, auth)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # construct the output file path
            output_file = os.path.join(target_dir, f"{q_id}_{counter}.json")

            # dump the JSON to the output file
            with open(output_file, "w") as file:
                json.dump(obj, file)

if __name__ == "__main__":
    with open("config.json", "r") as handler:
        info = json.load(handler)
    auth = (info["user"], info["pass"])

    download('Baseline_Jsons', 2000, True, auth)
    download('Selective_Jsons', 100, False, auth)