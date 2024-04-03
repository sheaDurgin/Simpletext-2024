from ranx import Qrels, Run, evaluate
import sys

def main():
    file_name = sys.argv[1]
    qrels = Qrels.from_file("simpletext_2023_task1_test.qrels", kind="trec")
    print(qrels.size)
    run = Run.from_file(file_name, kind="trec")

    mrr_score = evaluate(qrels, run, "mrr", make_comparable=True)
    ndcg_score = evaluate(qrels, run, "ndcg@10", make_comparable=True)
    map_score = evaluate(qrels, run, "map", make_comparable=True)
    print(f"MRR: {mrr_score}")
    print(f"NDCG@10: {ndcg_score}")
    print(f"MAP: {map_score}")

if __name__ == '__main__':
    main()
