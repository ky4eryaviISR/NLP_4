import codecs 
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load('en')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)


def read_lines(fname):
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-","(")
        sent = sent.replace("-RRB-",")")
        yield sent_id, sent


def parse_sen(sent_str):
    sent = nlp(sent_str)
    # print("#id:", sent_id)
    # print("#text:", sent.text)
    temp = []
    for word in sent:
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

def corpus_parse(file):
    sentence_dict = {}
    for sent_id, sent_str in read_lines(file):
        sentence_dict[sent_id] = parse_sen(sent_str)
    return sentence_dict
