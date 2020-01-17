import spacy
nlp = spacy.load('en')


def dep_mark_true(per, org, sent):
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["DEP"] == "mark":
            return True
    return False


def count_mark(per, org, sent):
    count = 0
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["DEP"] == "mark":
            count += 1
    return count


def pos_conj_true(per, org, sent):
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["POS"] == "CONJ":
            return True
        return False


def count_conj(per, org, sent):
    count = 0
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["POS"] == "CONJ":
            count += 1
    return count


def pos_verb_true(per, org, sent):
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["POS"] == "VERB":
            return True
    return False


def count_verb(per, org,sent):
    count = 0
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["POS"] == "VERB":
            count += 1
    return count


def pos_punct_true(per, org, sent):
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["POS"] == "PUNCT":
            return True
    return False


def count_punt(per, org, sent):
    count = 0
    for i in range(int(per['ID']) - 1, int(org['ID']) - 1):
        if sent[i]["POS"] == "PUNCT":
            count += 1
    return count


def count_org(sentence):
    count = 0
    for word in sentence:
        if word['TYPE'] == 'ORG':
            count += 1
    return count


def count_pers(sentence):
    count = 0
    for word in sentence:
        if word['TYPE'] == 'PERS':
            count += 1
    return


def list_lemma(per, org, sent): #List of lemmas between per and org
    start = int(per['ID']) - 1
    end = int(org['ID']) - 1
    words = []
    for i in range(min(start, end), max(start, end)):
        if len(sent[i]["LEMMA"]) > 0:
            words.append(sent[i]["LEMMA"])
    return list(set(words))


def list_pos(per, org, sent): #List of POS between per and org
    start = int(per['ID']) - 1
    end = int(org['ID']) - 1
    words = []
    for i in range(min(start, end), max(start, end)):
        if len(sent[i]["POS"]) > 0:
            words.append(sent[i]["POS"])
    return list(set(words))


def list_dep(per, org, sent): #List of DEP between per and org
    start = int(per['ID']) - 1
    end = int(org['ID']) - 1
    words = []
    for i in range(min(start, end), max(start, end)):
        if len(sent[i]["DEP"]) > 0:
            words.append(sent[i]["DEP"])
    return list(set(words))


def sentence_only(sent):
    return " ".join(sent["TEXT"])


def extraction(num, sent):
    pers = []
    orga = []
    new_sent = []
    sent_only = sentence_only(sent)
    parsed = nlp('unicode-escape')
    entities = map(lambda x: str(x.text), list(parsed.ents))

    for entity in entities:
        for word in sent:
            if word["TEXT"] == entity:
                pers.append(word)
                orga.append(word)

    for per in pers:
        for org in orga:
            if per["TEXT"] != org["TEXT"] and org['TYPE'] != 'PERS' and pers['TYPE'] == 'PERS':
                sentence = [num, per, org, sent]
                new_sent.append(sentence)

    return new_sent


keywords = {'work', 'head', 'serve', 'retire', 'found', 'star', 'conduct', 'transfer', 'direct', 'perform'}