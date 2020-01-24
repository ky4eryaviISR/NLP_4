import pickle
from datetime import datetime
from sys import argv
import numpy as np
import spacy
from sklearn.metrics import confusion_matrix, classification_report
from dataParser import Parser

nlp = spacy.load('en_core_web_lg')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)


class Classifier(object):
    def __init__(self, feat2id, model):
        self.model = pickle.load(open(model, 'rb'))
        self.f2id = feat2id

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

    def predict(self, sentence):
        res = []
        parsed = nlp(sentence)
        sen_parsed = Parser.load_to_dict(parsed)
        ner_dict = Parser.load_ner(parsed)
        ner_ent = Parser.build_ner_pair(ner_dict)
        sen_features = Parser.convert_sentence_2_feature(sen_parsed)
        features, labels = Parser.convert_ner_2_feature(ner_ent, sen_parsed)
        if len(sen_features) > 0:
            features = [line + sen_features for line in features]
        for sen, label in zip(features, labels):
            tmp = sen
            source = [i.split('=')[1] for i in sen if i is not None and i.startswith('f_ner')][0]
            target = [i.split('=')[1] for i in sen if i is not None and i.startswith('t_ner')][0]

            if target not in {'PERSON', 'ORG', 'None'} or source not in {'PERSON', 'DATE', 'ORG'} or \
                source+' '+target not in {'PERSON ORG', 'PERSON None', 'ORG ORG', 'PERSON PERSON', 'DATE ORG'}:
                y_hat = 0
            else:
                sen = self.get_vector(sen)
                y_hat = self.model.predict(sen.reshape(1, -1))[0]
                y_hat = bool(y_hat)
            if y_hat:
                res.append(label)
        return res


def evaluate(sen_f, f2id, model):
    classifier = Classifier(f2id, model)
    with open(argv[2], 'w') as fp:
        for line in open(sen_f):
            id_sen, sentence = line.split('\t', 1)
            res = classifier.predict(sentence)
            if len(res) > 0:
                for item in res:
                    fp.write(f'{id_sen}\t{item.strip()}\t({sentence.strip()})\n')


def main():
    print(datetime.now(), 'Load dictionary and load model')
    dict_vec = argv[3] if len(argv) > 3 else 'feat_dict'
    model = argv[4] if len(argv) > 4 else 'model'
    pred_file = argv[1]
    if pred_file.endswith('.processed'):
        print("Please use .txt file instead of .processed")
        return
    feat2id = {line.split('\t')[0]: int(line.split('\t', 1)[1]) for line in open(dict_vec).readlines()}
    print(datetime.now(), 'Starting Predict')
    evaluate(argv[1], feat2id,model)
    print(datetime.now(), 'Finished')


if __name__ == '__main__':
    main()
