from collections import Counter
import matplotlib.pyplot as plt
import spacy

from dataParser import LABEL, Parser
from extract_train import get_gold

nlp = spacy.load('en_core_web_lg')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

gold = get_gold('data/DEV.annotations')
ner_d = {}
for line in open('data/Corpus.DEV.txt'):
    id_sen, sentence = line.split('\t', 1)
    parsed = nlp(sentence)
    sen_parsed = Parser.load_to_dict(parsed)
    ner_dict = Parser.load_ner(parsed)
    ner_d[id_sen] = ner_dict


########################## FP investigate #########################

sentences = {}

for line in open('FP').readlines():
    sen_id, label = line.split(' ', 1)
    source, _, target = label.strip().split('\t')
    fr = ner_d[sen_id][source]['NER']
    tr = ner_d[sen_id][target]['NER']
    if sen_id not in sentences:
        sentences[sen_id] = []
    sentences[sen_id].append(fr+'_'+tr)

s = sentences
cnt = [i for s in sentences.values() for i in s]
print(s)
cnt = Counter(cnt)
print(cnt)
print(len(s))
plt.title("Evaluation on dev FP failure of model")
plt.xlabel("NER relation")
plt.ylabel("Failures")
plt.bar(list(cnt.keys()), cnt.values(), color='sandybrown')
plt.savefig("FP failure of model")
plt.show()

########################### FN investigate
source_lbl = []

for line in open('FN'):
    l = line.replace('\n', '')
    sen_id, value = l.split(' ', 1)
    source, label, target = value.split('\t')
    s = ner_d[sen_id][source]['NER'] if source in ner_d[sen_id] else 'None'
    t = ner_d[sen_id][target]['NER'] if target in ner_d[sen_id] else 'None'

    source_lbl.append(s+ ' '+t)
cnt = Counter(source_lbl)
plt.title("Evaluation on dev FN")
plt.xlabel("NER")
plt.ylabel("Failures")
plt.bar(list(cnt.keys()),cnt.values(), color='sandybrown')
plt.savefig("FN failure of model")
plt.show()
