import pickle
from datetime import datetime
from sys import argv
import numpy as np


from data.spc import corpus_parse, parse_sen
from ex_4 import get_sen_key_words, corpus_tags, get_ner_sen_pairs, get_vector, LABEL


class Classifier(object):
    def __init__(self, feat2id):
        self.model = pickle.load(open(argv[2],'rb'))
        self.f2id = feat2id

    def predict(self, sen):
        res = []
        key_words = get_sen_key_words(sen)
        sentence_pos_feat = get_ner_sen_pairs(sen)
        for sen in sentence_pos_feat:
            if len(key_words) > 0:
                sen += key_words
            vec_label = sen.pop(0)
            sen = get_vector(sen, self.f2id)
            y_hat = self.model.predict(sen.reshape(1, -1))[0]
            if y_hat == 1:
                res.append(vec_label)
        return res


def evaluate(sen_f, f2id):
    real = []
    pred = []
    classifier = Classifier(f2id)
    with open('result', 'w') as fp:
        for line in open(sen_f):
            id_sen, sentence = line.split('\t', 1)
            sen = parse_sen(sentence)
            res = classifier.predict(sen)
            if len(res) > 0:
                for item in res:
                    # print(f'{id_sen}\t{item}\t({sentence.strip()})')
                    fp.write(f'{id_sen}\t{item}\t({sentence.strip()})\n')
    return np.array(real), np.array(pred)

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
    expected_WorkFor = [item for item in real_dict if "Work_" in item]
    TP = len([i for i in pred_dict if i in expected_WorkFor])
    FN = len([i for i in pred_dict if i not in expected_WorkFor])
    FP = len([i for i in expected_WorkFor if i not in pred_dict])
    TN = len(real_dict) - TP - FP - FN
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*(precision*recall)/(precision+recall)
    print(f'{TP} {FP}\n{FN} {TN}\n')
    print(f'Precision={precision} recall={recall} F1={F1}')
    print('x')


def main():

    print(datetime.now(), 'Load dictionary and load model')
    feat2id = {line.split('\t')[0]: int(line.split('\t', 1)[1]) for line in open(argv[3]).readlines()}
    print(datetime.now(), 'Starting Validation')
    evaluate(argv[1], feat2id)
    calculate_accuracy(argv[4], 'result')
    # print(confusion_matrix(real, pred))


if __name__=='__main__':
    main()