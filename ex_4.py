from collections import Counter
import matplotlib.pyplot as plt
import spacy

from dataParser import LABEL, Parser
from extract_train import get_gold

gold = get_gold('data/DEV.annotations')

########################## FP investigate #########################

# sentences = {}
#
# for line in open('FP').readlines():
#     l = line.replace('\n', '')
#     if any([i for i in gold[l] if LABEL in i]):
#         continue
#     if l not in sentences:
#         sentences[l] = []
#     sentences[l].append(gold[l])
#
# s = sentences
# cnt = [j.split('\t')[1]  for s in sentences.values() for i in s for j in i]
# print(s)
# cnt = Counter(cnt)
# print(cnt)
# print(len(s))
# plt.title("False Positive failure of model")
# plt.xlabel("Label relation")
# plt.ylabel("Failures")
# plt.bar(list(cnt.keys()), cnt.values(), color='sandybrown')
# plt.savefig("FP failure of model")


########################### FN investigate

nlp = spacy.load('en_core_web_lg')
infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

ner_d = {}
for line in open('data/Corpus.TRAIN.txt'):
    id_sen, sentence = line.split('\t', 1)
    parsed = nlp(sentence)
    sen_parsed = Parser.load_to_dict(parsed)
    ner_dict = Parser.load_ner(parsed)
    ner_d[id_sen] = ner_dict


source_lbl = []

for line in open('FN'):
    l = line.replace('\n', '')
    sen_id, value = l.split(' ', 1)
    source, label, target = value.split('\t')
    s = ner_d[sen_id][source]['NER'] if source in  ner_d[sen_id] else 'None'
    t = ner_d[sen_id][target]['NER'] if target in ner_d[sen_id] else 'None'

    source_lbl.append(s+ ' '+t)
cnt = Counter(source_lbl)
plt.title("False Negative failure of Work_For NER")
plt.xlabel("NER")
plt.ylabel("Failures")
plt.bar(list(cnt.keys()),cnt.values(), color='sandybrown')
plt.show()
