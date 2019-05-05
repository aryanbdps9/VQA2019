import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from dataset import Dictionary


def dump_qns(dataroot, maxlen=14):
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    dump_files = [f[:-5]+'_dump.txt' for f in files]
    dump_idx_files = [f[:-5]+'_idx_dump.txt' for f in files]
    for ind, path in enumerate(files):
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        qsdpath = os.path.join(dataroot, dump_files[ind])
        qsidpath = os.path.join(dataroot, dump_idx_files[ind])
        qsd = open(qsdpath, 'w')
        qsidx = open(qsidpath, 'w')
        print("qsd:{}\tqsidx:{}".format(qsdpath, qsidpath))
        i = 0
        for q in qs:
            i += 1
            qstr = q['question'].strip()
            qsplit = qstr.split()[:maxlen]
            qstr_final = ' '.join(qsplit) + '\n'
            qid = str(q['question_id'])
            qsd.write(qstr_final)
            qsidx.write(qid+'\n')
            if i % 1000 == 0:
                print('i = {}'.format(i), end='\r')
        qsd.close()
    
dump_qns('../data', 14)
