import subprocess

prefix = ''

baseline_folder = f"{prefix}Baseline_Jsons/"
selective_folder = f"{prefix}Selective_Jsons/"

baseline_file = f"{prefix}baseline.txt"
selective_file = f"{prefix}selective.txt"
rr_baseline_file = f"{prefix}rr_baseline.txt"

n = "95"
model_path = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
ft_model_path = f"{prefix}5epochs-{model_path.split('/')[-1]}"
final_file = f"{prefix}final_results.txt"

print("Downloading all JSONS")
subprocess.run(["python", "download_jsons.py", baseline_folder, "2000", "1"])
subprocess.run(["python", "download_jsons.py", selective_folder, "100", "0"])

print("Creating Baseline txt")
subprocess.run(["python", "read_json.py", baseline_folder, baseline_file, n])
print("Creating Selective txt")
subprocess.run(["python", "read_json.py", selective_folder, selective_file])

print("Fine tuning cross encoder, with only train data")
subprocess.run(["python", "finetune.py", model_path])
# subprocess.run(["python", "finetune.py", model_path, "--final"])

print("Running cross encoder")
subprocess.run(["python", "cross_encoder.py", baseline_file, rr_baseline_file, baseline_folder, ft_model_path]) # rr = reranked

print("Combining scores of reranked baseline and selective")
subprocess.run(["python", "combine_scores.py", rr_baseline_file, selective_file, final_file])

print("Running evaluation")
subprocess.run(["python", "evaluation.py", final_file])

print("Fine tuning cross encoder, with all data")
# subprocess.run(["python", "finetune.py", model_path])
subprocess.run(["python", "finetune.py", model_path, "--final"])

print("Running cross encoder")
subprocess.run(["python", "cross_encoder.py", baseline_file, rr_baseline_file, baseline_folder, ft_model_path]) # rr = reranked

print("Combining scores of reranked baseline and selective")
subprocess.run(["python", "combine_scores.py", rr_baseline_file, selective_file, final_file])

print("Model trained on all data successfully")
