
LABEL = '\tWork_For\t'


class Parser(object):

    @staticmethod
    def build_vocabulary():
        vector_set = set()
        with open('feature_file') as fp:
            for line in fp:
                vector = line.strip().split(' ', 1)[1].split('/')
                vector_set.update(vector)
        feat2id = {v: k for k, v in enumerate(vector_set)}
        with open('feat_dict', 'w') as fp:
            fp.write('\n'.join([k + '\t' + str(v) for k, v in feat2id.items()]))
        return feat2id

    @staticmethod
    def build_feature_vec(ner):
        with open('feature_file', 'w') as fp:
            for _, value in ner.items():
                lbl, source, target, sen = value.values()
                ner_s = 'sourceNER=' + source['NER']
                ner_t = 'TargetNER=' + target['NER']
                if lbl == 'True':
                    fp.write("True " + ner_s + '/' + ner_t + '\n')
                else:
                    fp.write("False " + ner_s + '/' + ner_t + '\n')

    i = 1

    @staticmethod
    def build_ner_pair(ner, gold, sen):
        ner_pair = {}
        for n1, n1_val in ner.items():
            for n2, n2_val in ner.items():
                lbl = n1 + LABEL + n2
                if lbl in gold:
                    print(Parser.i)
                    Parser.i+=1
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

    @staticmethod
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

    @staticmethod
    def load_ner(parsed, sen_id):
        ner = {}
        for ent in parsed.ents:
            txt = ent.text
            if txt.endswith("'s"):
                index = txt.index('\'s')
                txt = txt[:index].strip()
            if txt.startswith('the '):
                txt = txt[4:]
            # print(txt)
            ner[txt] = {
                "NER": ent.root.ent_type_,
                "startText": ent.root.text,
                "startDep": ent.root.dep_,
                "startHead": ent.root.head.text,
                "id": sen_id
            }
        for ent in parsed.noun_chunks:
            txt = ent.text
            if txt.startswith('the '):
                txt = txt[4:]
                if txt.startswith('the '):
                    txt = txt[4:]

            if ',' in txt:
                index = txt.index(',')
                txt = txt[:index].strip()

            if txt.endswith("'s"):
                index = txt.index('\'s')
                txt = txt[:index].strip()
            # print(txt)
            ner[txt] = {
                "NER": ent.root.ent_type_ if ent.root.ent_type_ != '' else 'None',
                "startText": ent.root.text,
                "startText": ent.root.text,
                "startDep": ent.root.dep_,
                "startHead": ent.root.head.text,
                "id": sen_id
            }
        return ner
