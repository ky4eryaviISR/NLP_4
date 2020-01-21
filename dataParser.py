
LABEL = '\tWork_For\t'
KEYWORDS = {'work', 'head', 'serve', 'retire', 'found', 'star', 'conduct', 'transfer', 'direct', 'perform',
            'heads', 'former', 'AP', 'of', '\'s',
            'death', 'murder', 'assassinate', 'fire', 'shoot', 'members', 'director', 'employ', 'company'}
PHRASEWORDS = {'of the'}

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
    def write_2_file(feat, fp):

        for line in feat:
            lbl = line[0]
            vec = '/'.join(line[1:])
            if lbl == 'True':
                fp.write("True " + vec + '\n')
            else:
                fp.write("False " + vec + '\n')

    i = 1

    @staticmethod
    def build_ner_pair(ner, gold=None):
        ner_pair = {}
        for n1, n1_val in ner.items():
            for n2, n2_val in ner.items():
                lbl = n1 + LABEL + n2
                if n1 == n2:
                    continue
                if not gold:
                    ner_pair[lbl] = {'Label': None,
                                     'Source': n1_val,
                                     'Target': n2_val,
                                     }
                elif lbl in gold:
                    print(Parser.i)
                    Parser.i += 1
                    ner_pair[lbl] = {'Label': 'True',
                                     'Source': n1_val,
                                     'Target': n2_val,
                                     }
                else:
                    ner_pair[lbl] = {'Label': 'False',
                                     'Source': n1_val,
                                     'Target': n2_val,
                                     }
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
    def load_ner(parsed):
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
                "tag": ent.root.tag_,
                "id": ent.root.i
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
                "startDep": ent.root.dep_,
                "startHead": ent.root.head.text,
                "tag": ent.root.tag_,
                "id": ent.root.i
            }
        return ner

    @staticmethod
    def convert_sentence_2_feature(sen):
        phrase = []
        for key in PHRASEWORDS:
            for i, w in enumerate(sen):
                if w['lemma'] == key[0] and sen[i + 1]['lemma'] == key[1]:
                    phrase.append(key + '=True')
        return list(set([item['lemma'] + '=True' for item in sen if item['lemma'] in KEYWORDS] + phrase))

    @staticmethod
    def convert_ner_2_feature(ner_ent, sen):
        ners_features = []
        lbl_features = []
        for label, attr in ner_ent.items():
            isWork, source, target = attr['Label'], attr['Source'], attr['Target']
            f_word = 'f_word=' + source['startText']
            f_feat = 's_tag=' + source['tag']
            f_ner = 'f_ner=' + source['NER']
            n_tag = 'n_tag=' + (sen[source['id'] + 1]['tag'] if source['id'] + 1 < len(sen) else 'END')
            nn_tag = 'nn_tag=' + (sen[source['id'] + 2]['tag'] if source['id'] + 2 < len(sen) else 'END')

            t_word = 't_word=' + target['startText']
            t_feat = 't_tag=' + target['tag']
            t_ner = 't_ner=' + target['NER']
            p_tag = 'p_tag=' + (sen[target['id'] - 1]['tag'] if target['id'] - 1 >= 0 else 'START')
            pp_tag = 'pp_tag=' + (sen[target['id'] - 2]['tag'] if target['id'] - 2 >= 0 else 'START')
            dist = 'dist=' + str(target['id'] - source['id'])
            raw_feat = [isWork, f_word, f_ner, t_ner, t_word, f_feat, t_feat, n_tag, nn_tag, p_tag, pp_tag, dist]
            ners_features.append(raw_feat)
            lbl_features.append(label)
        return ners_features, lbl_features