with open('simpletext_2024_task1_train.qrels', 'r') as f1, open ('g_test.qrels', 'w') as f2:
    lines = f1.readlines()
    for line in lines:
        qid, _, _, _ = line.split(' ')
        if qid[0] == 'G' and int(qid[1:3]) > 15:
            f2.write(line)

with open('simpletext_2024_task1_train.qrels', 'r') as f1, open ('t_test.qrels', 'w') as f2:
    lines = f1.readlines()
    for line in lines:
        qid, _, _, _ = line.split(' ')
        if qid[0] == 'T':
            f2.write(line)

with open('simpletext_2024_task1_train.qrels', 'r') as f1, open ('g_and_t_test.qrels', 'w') as f2:
    lines = f1.readlines()
    for line in lines:
        qid, _, _, _ = line.split(' ')
        if qid[0] == 'G' and int(qid[1:3]) > 15:
            f2.write(line)
        elif qid[0] == 'T':
            f2.write(line)