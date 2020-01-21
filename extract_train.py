import pickle
from sys import argv

import spacy
from datetime import datetime

from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression


from spacy.tokenizer import Tokenizer

from dataParser import Parser

nlp = spacy.load('en_core_web_lg')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)


def get_gold(gold_f):
    gold = {}
    for line in open(gold_f):
        sen_id = line.split()[0]
        label = line.split('(')[0].split('\t', 1)[1].strip()
        if sen_id not in gold:
            gold[sen_id] = []
        gold[sen_id].append(label)
    return gold


def parse_corpus(train_f, is_train=False):
    ner_pair = {}
    gold = get_gold(argv[2])
    gold_lst = len(list([j for i in list(gold.values()) for j in i if 'Work_For' in j]))
    with open('feature_file', 'w') as fp:
        for line in open(train_f):
            sen_id, sen = line.split('\t')
            parsed = nlp(sen)
            sen_parsed = Parser.load_to_dict(parsed)
            ner_dict = Parser.load_ner(parsed)
            ner_ent = Parser.build_ner_pair(ner_dict, gold[sen_id])
            sen_features = Parser.convert_sentence_2_feature(sen_parsed)
            features, _ = Parser.convert_ner_2_feature(ner_ent, sen_parsed)
            if len(sen_features) > 0:
                features = [line + sen_features for line in features]
            Parser.write_2_file(features, fp)
    pred_ent = list(ner_pair.keys())
    print(f"Find good labels entities using spacy parser: {Parser.i}/{gold_lst}")
    if is_train:
        feat2id = Parser.build_vocabulary()
    else:
        feat2id = {line.split('\t')[0]: int(line.split('\t', 1)[1]) for line in open(argv[3]).readlines()}
    build_sparse_vectors(feat2id)


def build_sparse_vectors(f2id):
    with open('sparse', 'w') as fp:
        for line in open('feature_file'):
            label = line.split()[0]
            temp = ('1' if label == 'True' else '0')+" "
            temp_vec = []
            for vec in line.strip().split(' ', 1)[1].split('/'):
                temp_vec.append(f2id[vec.strip()])
            temp += ' '.join([str(i)+':1' for i in sorted(temp_vec)])
            fp.write(temp+'\n')


def train_model():
    x, y = load_svmlight_file('sparse', zero_based=True)
    print(datetime.now())
    model = LogisticRegression(multi_class='auto', solver='liblinear',
                               class_weight='balanced',
                               penalty='l1')
    model.fit(x, y)
    print(model.score(x, y))
    print(datetime.now())
    pickle.dump(model, open('model', 'wb'))
    return model


def main():
    parse_corpus(argv[1], is_train=True)
    train_model()


if __name__ == '__main__':
    main()
