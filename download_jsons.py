import requests
import csv
import json
import os
import sys
from tqdm import tqdm

url_prefix = ''
csv_file = 'simpletext_2024_task1_queries.csv'

def get_extra_results(obj, query_to_es, query, auth):
    remaining = 2000 - len(obj['hits']['hits'])
    remade_query = query_to_es.split("q=")[0] + "q=" + query + "&size=" + str(remaining)
    result = requests.get(remade_query, auth=auth).content.decode("utf-8")
    obj2 = json.loads(result)
    for item in obj2['hits']['hits']:
        if item not in obj['hits']['hits']:
            obj['hits']['hits'].append(item)

def download(target_dir, extra_results, num_of_results, auth):
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        reader_len = sum(1 for line in reader) - 1
        f.seek(0)
        next(reader)
        pre_qid = ""
        counter = 1
    
        for line in tqdm(reader, desc='Creating JSONS', total=reader_len):
            query_to_es = url_prefix + line[-1]
            url = query_to_es+ "&size=" + str(num_of_results)
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result = requests.get(url, auth=auth).content.decode("utf-8")
            obj = json.loads(result)
            
            hits_count = len(obj['hits']['hits'])
            if extra_results and hits_count < 2000:
                original_query = query_to_es.split("q=")[1][1:-1]
                if query_to_es.endswith("\""):
                    # remove quote
                    get_extra_results(obj, query_to_es, query_to_es.split("q=")[1][1:-1], auth)

                # replace with topic text
                get_extra_results(obj, query_to_es, line[1], auth)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # construct the output file path
            output_file = os.path.join(target_dir, f"{line[3].replace('.', '_')}.json")

            # dump the JSON to the output file
            with open(output_file, "w") as file:
                json.dump(obj, file)

if __name__ == "__main__":
    with open("config.json", "r") as handler:
        info = json.load(handler)
    auth = (info["user"], info["pass"])
    url_prefix = info["url"]

    args = sys.argv[1:]

    dir_name = args[0]
    num_of_results = int(args[1])
    extra_results = True if int(args[2]) == 1 else False

    download(dir_name, extra_results, num_of_results, auth)