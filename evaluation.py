from ranx import Qrels, Run, evaluate
import sys

def main():
    file_name = sys.argv[1]
    qrel_file = sys.argv[2]
    qrels = Qrels.from_file(qrel_file, kind="trec")
    print(qrels.size)
    run = Run.from_file(file_name, kind="trec")

    mrr_score = evaluate(qrels, run, "mrr", make_comparable=True)
    ndcg_score10 = evaluate(qrels, run, "ndcg@10", make_comparable=True)
    ndcg_score20 = evaluate(qrels, run, "ndcg@20", make_comparable=True)
    map_score = evaluate(qrels, run, "map", make_comparable=True)
    bpref_score = evaluate(qrels, run, "bpref", make_comparable=True)
    print(f"MRR: {mrr_score}")
    print(f"NDCG@10: {ndcg_score10}")
    print(f"NDCG@20: {ndcg_score20}")
    print(f"MAP: {map_score}")
    print(f"BPRREF: {bpref_score}")

if __name__ == '__main__':
    main()