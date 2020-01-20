import pickle
from sys import argv

import spacy
from datetime import datetime
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression

LABEL = '\tLive_In\t'
from spacy.tokenizer import Tokenizer

nlp = spacy.load('en_core_web_lg')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)


def load_to_dict(tokenized):
    temp = []
    for word in tokenized:
        head_id = str(word.head.i + 1)  # we want ids to be 1 based
        if word == word.head:  # and the ROOT to be 0.
            assert (word.dep_ == "ROOT"), word.dep_
            head_id = "0"  # root
        temp.append({
            'id': str(word.i + 1),
            'text': word.text,
            'lemma': word.lemma_,
            'tag': word.tag_,
            'pos': word.pos_,
            'head_id': head_id,
            'dep': word.dep_,
            'ent_iob': word.ent_iob_,
            'ent_type': word.ent_type_
        })
    return temp


def load_ner(parsed, sen_id):
    ner = {}
    for ent in parsed.ents:
        # print(ent.text)
        # if ent.text.endswith('.'):
        #     print(ent.text)
        ner[ent.text] = {
            "NER": ent.root.ent_type_,
            "startText": ent.root.text,
            "startDep": ent.root.dep_,
            "startHead": ent.root.head.text,
            "id": sen_id
        }
    for ent in parsed.noun_chunks:
        # print(ent.text)
        # if ent.text.endswith('.'):
        #     print(ent.text)
        if ent.text.startswith('the'):
            ent.text.replace('the ', '')
        txt = ent.text.rstrip(' ').rstrip('.')
        ner[txt] = {
            "NER": ent.root.ent_type_ if ent.root.ent_type_ !='' else 'None',
            "startText": ent.root.text,
            "startDep": ent.root.dep_,
            "startHead": ent.root.head.text,
            "id": sen_id
        }
    return ner


def get_gold(gold_f):
    gold = {}
    for line in open(gold_f):
        sen_id = line.split()[0]
        label = line.split('(')[0].split('\t', 1)[1].strip()
        if sen_id not in gold:
            gold[sen_id] = []
        gold[sen_id].append(label)
    return gold

i=1


def build_ner_pair(ner, gold, sen):
    global i
    ner_pair = {}
    for n1, n1_val in ner.items():
        for n2, n2_val in ner.items():
            lbl = n1+LABEL+n2
            if lbl in gold:
                print(i)
                i += 1
                ner_pair[lbl] = {'Label': 'True',
                                 'Source': n1_val,
                                 'Target': n2_val,
                                 'Sentence': sen}
            else:
                ner_pair[lbl] = {'Label': 'False',
                                 'Source': n1_val,
                                 'Target': n2_val,
                                 'Sentence': sen}
    return ner_pair


def build_feature_vec(ner):
    with open('feature_file', 'w') as fp:
        for _, value in ner.items():
            lbl, source, target, sen = value.values()
            ner_s = 'sourceNER='+source['NER']
            ner_t = 'TargetNER='+target['NER']
            if lbl == 'True':
                fp.write("True "+ner_s + '/' + ner_t + '\n')
            else:
                fp.write("False " + ner_s + '/' + ner_t + '\n')


def build_vocabulary():
    vector_set = set()
    with open('feature_file') as fp:
        for line in fp:
            vector = line.strip().split(' ', 1)[1].split('/')
            vector_set.update(vector)
    feat2id = {v: k for k, v in enumerate(vector_set)}
    with open('feat_dict', 'w') as fp:
        fp.write('\n'.join([k+'\t'+str(v) for k, v in feat2id.items()]))
    return feat2id


def parse_corpus(train_f,is_train=False):
    ner_pair = {}
    gold = get_gold(argv[2])
    for line in open(train_f):
        sen_id, sen = line.split('\t')
        parsed = nlp(sen)
        sen_parsed = load_to_dict(parsed)
        ner_dict = load_ner(parsed, sen_id)
        ner_pair.update(build_ner_pair(ner_dict, gold[sen_id], sen_parsed))
    build_feature_vec(ner_pair)
    if is_train:
        feat2id = build_vocabulary()
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
    model = LogisticRegression(multi_class='auto', solver='liblinear', class_weight='balanced', penalty='l1')
    model.fit(x, y)
    print(model.score(x, y))
    print(datetime.now())
    pickle.dump(model, open('model', 'wb'))
    return model



def main():
    parse_corpus(argv[1], is_train=True)
    train_model()


if __name__=='__main__':
    main()