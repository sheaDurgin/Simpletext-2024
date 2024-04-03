import subprocess

print("Downloading all JSONS")
subprocess.run(["python", "download_jsons.py", "Baseline_Jsons", "2000", "1"])
subprocess.run(["python", "download_jsons.py", "Selective_Jsons", "100", "0"])

print("Creating Baseline txt")
subprocess.run(["python", "read_json.py", "Baseline_Jsons", "baseline.txt"])
print("Creating Selective txt")
subprocess.run(["python", "read_json.py", "Selective_Jsons", "selective.txt"])

print("Running cross encoder")
subprocess.run(["python", "cross_encoder.py", "baseline.txt", "rr_baseline.txt"]) # rr = reranked

print("Combining scores of reranked baseline and selective")
subprocess.run(["python", "combine_scores.py", "rr_baseline.txt", "selective.txt", "final_results.txt"])

print("Running evaluation")
subprocess.run(["python", "evaluation.py", "final_results.txt"])