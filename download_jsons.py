import requests
import csv
import json
import os
import sys
from tqdm import tqdm
from Preprocessing_tools import *

url_prefix = ''
csv_file = 'simpletext_2024_task1_queries.csv'

def get_extra_results(obj, url_prefix, query, auth, num_of_results):
    remaining = num_of_results - len(obj['hits']['hits'])
    remade_query = url_prefix + query + "&size=" + str(remaining)
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
    
        for line in tqdm(reader, desc='Creating JSONS', total=reader_len):
            url = url_prefix + "\"" + line[-1] + "\"" + "&size=" + str(num_of_results)
            result = requests.get(url, auth=auth).content.decode("utf-8")
            obj = json.loads(result)
            
            hits_count = len(obj['hits']['hits'])
            if extra_results and hits_count < num_of_results:
                # remove quote
                get_extra_results(obj, url_prefix, line[-1], auth, num_of_results)

                # replace with topic text
                get_extra_results(obj, url_prefix, line[1], auth, num_of_results)

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