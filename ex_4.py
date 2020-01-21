from sys import argv
import numpy as np
from datetime import datetime

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from data.spc import corpus_parse
import pickle
from sklearn.metrics import confusion_matrix

LABEL = '\tWork_For\t'
KEYWORDS = {'work', 'head', 'serve', 'retire', 'found', 'star', 'conduct', 'transfer', 'direct', 'perform',
            'heads', 'former', 'AP', 'of', '\'s',
            'death', 'murder', 'assassinate', 'fire', 'shoot', 'members', 'director', 'employ', 'company'}
PHRASEWORDS = {'of the'}
BADPARSE = ['Soprano', 'Secretary-General']


def get_ner_sen_pairs(sen):
    ner_ent = []
    sen_features = []
    i = 0
    while i < len(sen):
        if sen[i]['ent_type'] != '' \
                and sen[i]['ent_iob'] in ['B', 'I'] \
                and sen[i]['tag'] != 'IN' \
                and sen[i]['text'] not in ['the', '\'s'] + BADPARSE:
            temp = [i]
            for j in range(i+1, len(sen)):
                if not (sen[j]['ent_iob'] in ['I', 'B']
                        or (sen[j]['pos'] == 'PROPN' and not sen[j]['text'].endswith('.'))
                        or ((j+1 < len(sen) and sen[j+1]['dep'] == 'pobj'
                             or sen[j]['dep'] == 'dobj')
                        and (j+1 <len(sen) and sen[j+1]['ent_iob'] != 'B'))
                    ) \
                    or (sen[j]['lemma'] == 'of' and sen[j+1]['lemma'] == 'the') \
                    or sen[j]['text'] == '\'s'\
                    or sen[j]['text'] == 'for'\
                        or sen[j]['text'] == ','\
                        or sen[j]['text'] in BADPARSE:
                    break
                temp.append(j)
            if i+1 >= len(sen):
                break
            i = j
            ner_ent.append(temp)
        else:
            i += 1
    for wp in ner_ent:
        from_ph = ' '.join([sen[i]['text'] for i in wp])
        f_word = 'f_word=' + from_ph
        f_feat = 's_tag=' + sen[wp[0]]['tag']
        f_ner = 'f_ner=' + sen[wp[0]]['ent_type']
        n_tag = 'n_tag=' + (sen[wp[0]+1]['tag'] if wp[0]+1 < len(sen) else 'END')
        nn_tag = 'nn_tag=' + (sen[wp[0]+2]['tag'] if wp[0]+2 < len(sen) else 'END')
        for to_wp in ner_ent:
            to_ph = ' '.join([sen[i]['text'] for i in to_wp])
            if from_ph == to_ph:
                continue
            t_word = 't_word=' + to_ph
            label = from_ph + LABEL + to_ph
            t_feat = 't_tag=' +sen[to_wp[0]]['tag']
            t_ner = 't_ner=' + sen[to_wp[0]]['ent_type']
            p_tag = 'p_tag=' + (sen[to_wp[0]]['tag'] if to_wp[0]-1 >= 0 else 'START')
            pp_tag = 'pp_tag=' + (sen[to_wp[0]]['tag'] if to_wp[0]-2 >= 0 else 'START')
            dist = 'dist=' + str(to_wp[0] - wp[0])
            raw_feat = [label, f_word, f_ner, t_ner, t_word,  f_feat, t_feat, n_tag, nn_tag, p_tag, pp_tag, dist]
            sen_features.append(raw_feat)
    return sen_features


def get_sen_key_words(sen):
    phrase = []
    for key in PHRASEWORDS:
        for i, w in enumerate(sen):
            if w['lemma'] == key[0] and sen[i+1]['lemma']== key[1]:
                phrase.append(key+'=True')
    return list(set([item['lemma']+'=True' for item in sen if item['lemma'] in KEYWORDS] + phrase))


def corpus_tags(f):
    sen = {}
    for line in open(f):
        sen_id = line.split()[0]
        label = line.split('(')[0].split('\t', 1)[1].strip()
        if sen_id not in sen:
            sen[sen_id] = [label]
        else:
            sen[sen_id].append(label)
    return sen


def get_corpus_features(sen_f, tag_f,file_name='feature_file'):
    sentence_dict = corpus_parse(sen_f)
    load_tags = corpus_tags(tag_f)
    count_p = count_n = 0
    with open(file_name, 'w') as fp:
        for sentence in sentence_dict.items():
            id_sen, sen = sentence
            print(id_sen)
            key_words = get_sen_key_words(sen)
            sen_tags = [tag for tag in load_tags[id_sen] if LABEL in tag]
            len_t = len(sen_tags)
            sentence_pos_feat = get_ner_sen_pairs(sen)
            for sen_f in sentence_pos_feat:
                if len(key_words) > 0:
                    sen_f += key_words
                label = sen_f.pop(0)
                sen_f = '/'.join([i for i in sen_f])
                if any([i for i in sen_tags if i == label]):
                    fp.write(f'True {sen_f}\n')
                    count_p += 1
                    len_t -=1
                else:
                    # if len(sen_tags)>0:
                    #     print(label)
                    #     print('/'.join(sen_tags)+'\n')
                    fp.write(f'False {sen_f}\n')
                    count_n += 1
            if len_t > 0:
                print("Not all")
    print(f"Count pos={count_p}, Count neg={count_n}")


def build_vocabulary():
    vector_set = set()
    with open('feature_file') as fp:
        for line in fp:
            vector = line.strip().split(' ', 1)[1].split('/')
            vector_set.update(vector)
    feat2id = {v: k for k, v in enumerate(vector_set)}
    id2feat = {k: v for k, v in enumerate(vector_set)}
    with open('feat_dict', 'w') as fp:
        fp.write('\n'.join([k+'\t'+str(v) for k, v in feat2id.items()]))
    return feat2id




def build_sparse_vectors(f2id):
    with open('sparse', 'w') as fp:
        for line in open('feature_file'):
            label = line.split()[0]
            temp = ('1' if label == 'True' else '0')+" "
            temp_vec = []
            for vec in line.split('/')[1:]:
                temp_vec.append(f2id[vec.strip()])
            temp += ' '.join([str(i)+':1' for i in sorted(temp_vec)])
            fp.write(temp+'\n')


def train_model():
    x, y = load_svmlight_file('sparse', zero_based=True)
    print(datetime.now())
    model = LogisticRegression(multi_class='auto', solver='liblinear', class_weight='balanced', penalty='l1', C=3)
    model.fit(x, y)
    print(model.score(x, y))
    print(datetime.now())
    pickle.dump(model, open('model','wb'))
    return model


def get_vector(sen_f,f2id):
    vec = np.zeros(len(f2id))
    for s in sen_f:
        if s in f2id:
            vec[f2id[s]] = 1
    return vec


def evaluate(sen_f, tag_f, f2id, model):
    real = []
    pred = []
    sentence_dict = corpus_parse(sen_f)
    load_tags = corpus_tags(tag_f)
    for sentence in sentence_dict.items():
        id_sen, sen = sentence
        key_words = get_sen_key_words(sen)
        sentence_pos_feat = get_ner_sen_pairs(sen)
        for sen in sentence_pos_feat:
            if len(key_words) > 0:
                sen += key_words
            vec_label = sen.pop(0)
            y = int(load_tags[id_sen] and vec_label in load_tags[id_sen])
            sen = get_vector(sen, f2id)
            y_hat = model.predict(sen.reshape(1, -1))[0]
            real.append(y)
            pred.append(y_hat)
            if load_tags[id_sen] == vec_label:
                print(load_tags[id_sen])
    return np.array(real), np.array(pred)


def main():
    print(datetime.now(), 'Build features')
    get_corpus_features(argv[1], argv[2])
    print(datetime.now(), 'Build vocabulary')
    feat2id = build_vocabulary()
    print(datetime.now(), 'Create sparse vectors')
    build_sparse_vectors(feat2id)
    print(datetime.now(), 'Starting Train')
    model = train_model()
    print(datetime.now(), 'Evaluate on Train')
    real, pred = evaluate(argv[1], argv[2], feat2id, model)
    print(confusion_matrix(real, pred))
    # print(datetime.now(), 'Starting Validation')
    # evaluate(argv[3], argv[4], feat2id, model)
    # print(confusion_matrix(real, pred))


if __name__ == '__main__':
    main()