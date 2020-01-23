# target label
LABEL = '\tWork_For\t'
# key words for feature vector
KEYWORDS = {'work', 'head', 'serve', 'retire', 'found', 'star', 'conduct', 'transfer', 'direct', 'perform',
            'shoot', 'assassinate', 'assassination', 'assassin', '\'', '"',
            'heads', 'former', 'AP', '\'s',
            'death', 'murder', 'fire', 'members', 'director', 'employ', 'company', 'investigate', 'kill', 'involve', 'gunman', 'hang', 'claim'}


class Parser(object):
    """
    class for data manipulation
    """
    gold_ent = 0

    @staticmethod
    def build_vocabulary():
        """
        running over feature file and creating another file with mapping feature
        to the integer for future use
        :return:
        """
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
        """
        getting feature vector and save it to file
        :param feat: feature vector
        :param fp: pointer to file
        :return:
        """
        for line in feat:
            lbl = line[0]
            vec = '/'.join(line[1:])
            if lbl == 'True':
                fp.write("True " + vec + '\n')
            else:
                fp.write("False " + vec + '\n')

    @staticmethod
    def build_ner_pair(ner, gold=None):
        """
        getting all ner entities and return the pairs
        :param ner: ner entities
        :param gold: possible gold entities passed while training only
        :return:
        """
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
                    Parser.gold_ent += 1
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
        """
        :param tokenized:get tokenized sentence
        :return: list of dictionaries for each word
        """
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
        """
        :param parsed: parsed sentence
        :return: parsed ner/noun chunks
        """
        ner = {}
        for ent in parsed.ents:
            txt = ent.text
            if txt.endswith("'s"):
                index = txt.index('\'s')
                txt = txt[:index].strip()
            if txt.startswith('the '):
                txt = txt[4:]
            ner[txt] = {
                "NER": ent.root.ent_type_,
                "startText": ent.root.text,
                "startDep": ent.root.dep_,
                "startHead": ent.root.head.text,
                "tag": ent.root.tag_,
                "id_start": ent.start,
                "id_end": ent.end,
                "text": txt.strip()
            }
        for ent in parsed.noun_chunks:
            txt = ent.text
            if txt.startswith('the '):
                txt = txt[4:]
                if txt.startswith('the '):
                    txt = txt[4:]

            if txt.endswith(','):
                index = txt.index(',')
                txt = txt[:index].strip()

            if txt.endswith("'s"):
                index = txt.index('\'s')
                txt = txt[:index].strip()
            ner[txt] = {
                "NER": ent.root.ent_type_ if ent.root.ent_type_ != '' else 'None',
                "startText": ent.root.text,
                "startDep": ent.root.dep_,
                "startHead": ent.root.head.text,
                "tag": ent.root.tag_,
                "id_start": ent.start,
                "id_end": ent.end,
                "text": txt.strip()
            }
        return ner

    @staticmethod
    def convert_sentence_2_feature(sen):
        """
        convert sentence to feature using key words
        :param sen: sentence
        :return: converted to feature
        """
        other = []
        for i, w in enumerate(sen):
            if w['ent_type'] == 'CD' and 'CD=True' not in other:
                other.append('CD=True')
        return list(set([item['lemma'] + '=True' for item in sen if item['lemma'] in KEYWORDS] + other))

    @staticmethod
    def convert_ner_2_feature(ner_ent, sen):
        """
        :param ner_ent: ner entity
        :param sen: sentence
        :return: features for each pair of entities
        """
        ners_features = []
        lbl_features = []
        for label, attr in ner_ent.items():
            isWork, source, target = attr['Label'], attr['Source'], attr['Target']
            if source['NER']+' '+target['NER'] not in {'PERSON ORG', 'PERSON None', 'ORG ORG', 'PERSON PERSON', 'DATE ORG'}:
                continue
            f_txt = 'f_txt=' + source['text']
            f_word = 'f_word=' + source['startText']
            f_feat = 'f_tag=' + source['tag']
            f_ner = 'f_ner=' + source['NER']
            f_dep = 'f_dep=' + source['startDep']
            n_dep = 'n_dep=' + (sen[source['id_end']]['dep'] if source['id_end'] < len(sen) else 'END')
            nn_dep = 'nn_dep=' + (sen[source['id_end']+1]['dep'] if source['id_end'] + 1 < len(sen) else 'END')
            n_tag = 'n_tag=' + (sen[source['id_end']]['tag'] if source['id_end'] < len(sen) else 'END')
            nn_tag = 'nn_tag=' + (sen[source['id_end'] + 1]['tag'] if source['id_end'] + 1 < len(sen) else 'END')

            t_txt = 't_txt=' + target['text']
            t_word = 't_word=' + target['startText']
            t_feat = 't_tag=' + target['tag']
            t_ner = 't_ner=' + target['NER']
            t_dep = 't_dep=' + target['startDep']
            p_dep = 'p_dep=' + (sen[target['id_start'] - 1]['dep'] if target['id_start'] - 1 >= 0 else 'START')
            pp_dep = 'pp_dep=' + (sen[target['id_start'] - 2]['dep'] if target['id_start'] - 2 >= 0 else 'START')
            p_tag = 'p_tag=' + (sen[target['id_start'] - 1]['tag'] if target['id_start'] - 1 >= 0 else 'START')
            pp_tag = 'pp_tag=' + (sen[target['id_start'] - 2]['tag'] if target['id_start'] - 2 >= 0 else 'START')
            dist = 'dist=' + str(target['id_start'] - source['id_start'])
            raw_feat = [isWork,
                        f_txt,t_txt,
                        f_word, f_ner, t_ner, t_word, f_feat, t_feat, n_tag,
                        nn_tag, p_tag, pp_tag, dist, f_dep, t_dep, n_dep, nn_dep, p_dep, pp_dep]
            ners_features.append(raw_feat)
            lbl_features.append(label)
        return ners_features, lbl_features
