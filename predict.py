import pickle
from datetime import datetime
from sys import argv
import numpy as np
import spacy
from sklearn.metrics import confusion_matrix, classification_report
from dataParser import Parser, LABEL

nlp = spacy.load('en_core_web_lg')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)


class Classifier(object):
    def __init__(self, feat2id, gold=None):
        self.model = pickle.load(open(argv[2], 'rb'))
        self.f2id = feat2id
        self.real_values = []
        self.predicted = []
        if gold:
            self.gold = self.load_gold(gold)

    def load_gold(self, gold_f):
        gold = {}
        for line in open(gold_f):
            sen_id = line.split()[0]
            label = line.split('(')[0].split('\t', 1)[1].strip()
            if sen_id not in gold:
                gold[sen_id] = []
            gold[sen_id].append(label)
        return gold


    def get_vector(self, sen_f):
        vec = np.zeros(len(self.f2id))
        for s in sen_f:
            if s in self.f2id:
                vec[self.f2id[s]] = 1
        return vec

    def predict(self, sen, sen_id, with_gold=False):

        res = []
        parsed = nlp(sen)
        sen_parsed = Parser.load_to_dict(parsed)
        ner_dict = Parser.load_ner(parsed)
        ner_ent = Parser.build_ner_pair(ner_dict)
        sen_features = Parser.convert_sentence_2_feature(sen_parsed)
        features, labels = Parser.convert_ner_2_feature(ner_ent, sen_parsed)
        if len(sen_features) > 0:
            features = [line + sen_features for line in features]
        for sen, label in zip(features, labels):
            sen = self.get_vector(sen)
            y_hat = self.model.predict(sen.reshape(1, -1))[0]
            if y_hat == 1:
                res.append(label)
            if with_gold:
                real = self.get_gold(label, sen_id)
                self.real_values.append(real)
                self.predicted.append(bool(y_hat))
        return res

    def get_gold(self,lbl, sen_id):
        return any([i for i in self.gold[sen_id] if lbl==i])

    def get_score(self):
        confusion_matrix(self.real_values, self.predicted)
        print(classification_report(self.real_values, self.predicted))



def evaluate(sen_f, f2id, gold):
    classifier = Classifier(f2id, gold)
    with open('result', 'w') as fp:
        for line in open(sen_f):
            id_sen, sentence = line.split('\t', 1)
            res = classifier.predict(sentence, id_sen, (gold is not None))
            if len(res) > 0:
                for item in res:
                    # print(f'{id_sen}\t{item}\t({sentence.strip()})')
                    fp.write(f'{id_sen}\t{item}\t({sentence.strip()})\n')
    classifier.get_score()


def main():
    print(datetime.now(), 'Load dictionary and load model')
    feat2id = {line.split('\t')[0]: int(line.split('\t', 1)[1]) for line in open(argv[3]).readlines()}
    print(datetime.now(), 'Starting Validation')
    gold = argv[4] if len(argv) > 4 else None
    evaluate(argv[1], feat2id, gold)


if __name__ == '__main__':
    main()
