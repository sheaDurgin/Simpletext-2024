import subprocess

prefix = ''

baseline_folder = f"{prefix}Baseline_Jsons/"
selective_folder = f"{prefix}Selective_Jsons/"

baseline_file = f"{prefix}baseline.txt"
selective_file = f"{prefix}selective.txt"
rr_baseline_file = f"{prefix}rr_baseline.txt"

n = "100"
epoch = "5"
lr = "1e-05"

model_path = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
ft_model_path = f"{prefix}final-{model_path.split('/')[-1]}"
final_file = f"{prefix}final_results.txt"

print("Downloading all JSONS")
subprocess.run(["python", "download_jsons.py", baseline_folder, "2000", "1"])
subprocess.run(["python", "download_jsons.py", selective_folder, "100", "0"])

print("Creating Baseline txt")
subprocess.run(["python", "read_json.py", baseline_folder, baseline_file, n])
print("Creating Selective txt")
subprocess.run(["python", "read_json.py", selective_folder, selective_file])

print("Creating all test qrels")
subprocess.run(["python", "create_test_qrels.py"])

# FINETUNE ON ALL EXCEPT 2023 TEST TOPICS
print("Fine tuning cross encoder on all data")
subprocess.run(["python", "finetune.py", model_path, epoch, lr])

print("Running cross encoder")
subprocess.run(["python", "cross_encoder.py", baseline_file, rr_baseline_file, baseline_folder, ft_model_path]) # rr = reranked

print("Combining scores of reranked baseline and selective")
subprocess.run(["python", "combine_scores.py", rr_baseline_file, selective_file, final_file])

print("Running evaluation on unseen G test")
subprocess.run(["python", "evaluation.py", final_file, 'g_test.qrels'])

print("Running evaluation on unseen T test")
subprocess.run(["python", "evaluation.py", final_file, 't_test.qrels'])

print("Running evaluation on unseen G and T test data")
subprocess.run(["python", "evaluation.py", final_file, 'g_and_t_test.qrels'])


# FINAL MODEL
print("Fine tuning cross encoder on all data")
subprocess.run(["python", "finetune.py", model_path, epoch, lr, "--final"])

print("Running cross encoder")
subprocess.run(["python", "cross_encoder.py", baseline_file, rr_baseline_file, baseline_folder, ft_model_path]) # rr = reranked

print("Combining scores of reranked baseline and selective")
subprocess.run(["python", "combine_scores.py", rr_baseline_file, selective_file, final_file])

print("Running evaluation on seen data")
subprocess.run(["python", "evaluation.py", final_file, 'g_and_t_test.qrels'])