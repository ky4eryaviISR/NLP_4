from data.spc import sentence_dict
from sklearn import svm


def get_ner_sen_pairs(sen):
    ner_ent = []
    for i, w_dict in enumerate(sen):
        if w_dict['ent_type'] != '' and w_dict['ent_iob'] == 'B':
            if len(sen) >= i and sen[i+1]['ent_iob'] == 'I':
                ner_ent.append((i, i+1))
            ner_ent.append(i)
    for ner in ner_ent:
        if isinstance(ner, tuple):
            print(sen[ner[0]]['text'], sen[ner[1]]['text'])
        else:
            print(sen[ner]['text'])
    return ner_ent


def main():
    print(sentence_dict)
    for sentence in sentence_dict.items():
        id_sen, sen = sentence
        print(id_sen)
        pairs = get_ner_sen_pairs(sen)


    model = svm.SVC()

    # model.fit()






if __name__ == '__main__':
    main()