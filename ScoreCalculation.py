from sys import argv

from dataParser import LABEL

def same(x,lst):
    return x in lst


def almost_same(x, lst):
    print(x)
    sen_id, x = x.split(' ', 1)
    source, label, target = x.split('\t')
    for item in lst:
        gold_id, x = item.split(' ',1)
        if gold_id!=sen_id:
            continue
        gold_s, gold_l,gold_t = x.split('\t')
        if (gold_s in source or source in gold_s) and (gold_t in target or target in gold_t):
            return True
    return False


def calculate_accuracy(real, predicted):
    real_dict = []
    pred_dict = []
    for line in open(real):
        sen_id = line.split()[0]
        label = line.split('(')[0].split('\t', 1)[1].strip()
        real_dict.append(sen_id + ' ' + label)
    for line in open(predicted):
        sen_id = line.split()[0]
        label = line.split('(')[0].split('\t', 1)[1].strip()
        pred_dict.append(sen_id + ' ' + label)
    expected_WorkFor = [item for item in real_dict if LABEL in item]
    TP = len([i for i in pred_dict if almost_same(i, expected_WorkFor)])
    FP = len([i for i in pred_dict if i not in expected_WorkFor])
    FN = len([i for i in expected_WorkFor if i not in pred_dict])
    TN = len(real_dict) - TP - FP - FN
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*(precision*recall)/(precision+recall)
    print(f'{TP} {FN}\n{FP} {TN}\n')
    print(f'Precision={precision} recall={recall} F1={F1}')
    print('x')


if __name__=='__main__':
    calculate_accuracy(argv[1], argv[2])
