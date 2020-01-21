from sys import argv

from dataParser import LABEL


def almost_same(x, lst):
    sen_id, x = x.split(' ', 1)
    return any([item for item in lst if sen_id in item and (x in item or item in x)])


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


# calculate_accuracy(argv[1], argv[2])
