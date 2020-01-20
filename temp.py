import sys
import pickle
import spacy
import codecs

import numpy as np
from scipy import sparse
from sklearn import metrics, svm, ensemble
from feature_builders import FeatureBuilders
from datetime import datetime

startTime = datetime.now()


def passedTime():
    return str(datetime.now() - startTime)


TAGS = {'OrgBased_In', 'Live_In', 'Kill', 'Located_In', 'Work_For'}
# TAGS = {'Live_In'}

NO_CONNECTION = 'None'

nlp = spacy.load('en')


class Model:
    def __init__(self):
        self.features = {}
        # self.clf = ensemble.RandomForestClassifier()
        self.clf = svm.LinearSVC()

    def train(self, data):
        features, tags = self.extractFeatures(data, True)
        self.clf.fit(features, tags)

    def predict(self, data):
        features, tags = self.extractFeatures(data, False)
        return self.clf.predict(features)

    def save(self, modelFile):
        pickle.dump((self.clf, self.features), open(modelFile, 'wb'))

    def load(self, modelFile):
        return 1  # TODO load a model from pickle

    def extractFeatures(self, data, isTrain):
        """
        extract Features
        :param data: data
        :param isTrain: is_train flag (bool)
        :return: features, tags
        """
        # allVectorFeatures = []
        allFeatures = []
        allTags = []
        for ((arg1, arg2, sentence), tag) in data:
            features = []
            vectorFeatures = []
            for feature_builder in FeatureBuilders.ALL:
                # vectorFeatures += feature_builder.build_vector(arg1, arg2, sentence)
                for feature in feature_builder.build_features(arg1, arg2, sentence):
                    if feature not in self.features and isTrain:
                        self.features[feature] = len(self.features)
                    if feature in self.features:
                        features.append(self.features[feature])

            # allVectorFeatures.append(vectorFeatures)
            allFeatures.append(features)
            allTags.append(tag)

        # X = sparse.csr_matrix([self.dense2sparse(dense) + allVectorFeatures[i] for i, dense in enumerate(allFeatures)])
        X = sparse.csr_matrix([self.dense2sparse(dense) for dense in allFeatures])
        Y = np.array(allTags)
        return X, Y

    def dense2sparse(self, dense):
        sparse = np.zeros(len(self.features))
        for i in dense:
            sparse[i] = 1
        return sparse


class Data:
    def __init__(self, trainFile, trainAnnotations = None):
        self.filters = self.entities = []
        self.trainFile = trainFile
        self.trainAnnotations = trainAnnotations

        self.corpus = self.readCorpus()
        self.annotations = self.readAnnotations()

        print "Creating data", passedTime()
        self.data = self.createData()
        # self.filterData()

        print "Generated", len(self.data), "data items", passedTime()
        print "Positive", len([0 for d, t in self.data if t in TAGS]), "/", \
            sum([len(c) for (i, c) in self.annotations.items()])

    # def isNoTag(self, ent1, ent2):
    #     isBoth = (ent1 + "-" + ent2) in {'PERSON-NORP', 'PERSON-GPE', 'ORG-GPE', 'PERSON-LOC', 'PERSON-ORG',
    #                                      'PERSON-PERSON', 'PRODUCT-GPE', 'LOC-GPE', 'GPE-GPE', 'ORG-LOC'}
    #     isOne = ent1 == "UNKNOWN" and ent2 in {"NORP", "GPE", "LOC", "ORG", "PERSON"} or \
    #             ent2 == "UNKNOWN" and ent1 in {"PERSON"}
    #     if CONNECTION_TYPE == "Work_For":
    #         isBoth = (ent1 + "-" + ent2) in {'PERSON-ORG', 'PERSON-GPE', 'PERSON-PERSON', 'ORG-ORG', 'GPE-ORG',
    #                                          'DATE-ORG'}
    #         isOne = ent1 == "UNKNOWN" and ent2 in {"ORG", "GPE", "PERSON"} or \
    #                 ent2 == "UNKNOWN" and ent1 in {"PERSON", "ORG", "GPE", "DATE", "UNKNOWN"}
    #     return not (isBoth or isOne)

    def filterData(self):
        """
        data filter
        :return: N/A
        """
        self.filters = set()
        workSet = set()
        for i in range(2):
            for data, tag in self.data:
                arg1, arg2, sentence = data
                name = arg1["entType"] + "-" + arg2["entType"]
                if i == 0 and tag == NO_CONNECTION:
                    self.filters.add(name)
                elif i == 1 and tag in TAGS:
                    workSet.add(name)
                    if name in self.filters:
                        self.filters.remove(name)

        # print self.filters
        # print workSet

        print "Total relations:", len(self.data)
        print "Positive", len([(d, t) for d, t in self.data if t != NO_CONNECTION])

        # for ((arg1, arg2, sentence), tag) in self.data:
        #     if tag != NO_CONNECTION and self.isNoTag(arg1[1], arg2[1]):
        #         print "--------"
        #         print "--------"
        #         print tag, arg1[1], arg2[1]
        #         print "--------"
        #         print "--------"

        self.data = filter(lambda ((arg1, arg2, sentence), tag): not self.isNoTag(arg1["entType"], arg2["entType"]),
                           self.data)
        print "Filtered relations:", len(self.data)
        print "Positive", len([(d, t) for d, t in self.data if t != NO_CONNECTION])

        self.data = filter(
            lambda ((arg1, arg2, sentence), tag): arg1["text"][0].isupper() and arg2["text"][0].isupper(),
            self.data)
        print "Remove non-uppercase relations:", len(self.data)
        print "Positive", len([(d, t) for d, t in self.data if t != NO_CONNECTION])

    def clean(self, text):
        return text.rstrip(".")

    def createData(self):
        """
        creates data based on spacy
        :return: data
        """
        data = []
        for id, sentence in self.corpus.items():
            parsed = nlp(sentence)
            sentenceData = []

            actualSentence = ""
            lCounter = 0
            sentenceDic = {}

            for i, word in enumerate(parsed):
                head_id = word.head.i + 1  # we want ids to be 1 based
                if word == word.head:  # and the ROOT to be 0.
                    assert (word.dep_ == "ROOT"), word.dep_
                    head_id = 0  # root

                sentenceData.append({
                    "id": word.i + 1,
                    "word": word.text,
                    "lemma": word.lemma_,
                    "pos": word.pos_,
                    "tag": word.tag_,
                    "parent": head_id,
                    "dependency": word.dep_,
                    "bio": word.ent_iob_,
                    "ner": word.ent_type_
                })
                actualSentence += " " + word.text
                sentenceDic[lCounter + 1] = i
                lCounter += 1 + len(word.text)

            entities = {}
            for entity in parsed.ents:
                cleanText = self.clean(entity.text)
                entities[cleanText] = {
                    "text": cleanText,
                    "originalText": entity.text,
                    "entType": entity.root.ent_type_,
                    "rootText": entity.root.text,
                    "rootDep": entity.root.dep_,
                    "rootHead": entity.root.head.text,
                    "id": id
                }

            for chunk in parsed.noun_chunks:
                cleanText = self.clean(chunk.text)
                if cleanText not in entities:
                    entities[cleanText] = {
                        "text": cleanText,
                        "originalText": chunk.text,
                        "entType": u'UNKNOWN',
                        "rootText": chunk.root.text,
                        "rootDep": chunk.root.dep_,
                        "rootHead": chunk.root.head.text,
                        "id": id
                    }

            for entity in entities.values():
                firstSentenceIndex = actualSentence.find(entity["originalText"])
                firstWordIndex = sentenceDic[firstSentenceIndex] if firstSentenceIndex in sentenceDic else 0

                lastSentenceIndex = firstSentenceIndex + len(entity["originalText"]) + 1
                lastWordIndex = sentenceDic[lastSentenceIndex] - 1 if lastSentenceIndex in sentenceDic else 0

                depIndex = actualSentence.find(entity["rootDep"])
                depWordIndex = sentenceDic[depIndex] if depIndex in sentenceDic else 0

                headIndex = actualSentence.find(entity["rootHead"])
                headWordIndex = sentenceDic[headIndex] if headIndex in sentenceDic else 0

                entity["firstWordIndex"] = firstWordIndex
                entity["lastWordIndex"] = lastWordIndex
                entity["headWordTag"] = sentenceData[headWordIndex]["tag"]
                entity["depWordIndex"] = depWordIndex

            self.entities = entities.values()

            isAnyAdded = False
            for ne1 in self.entities:
                for ne2 in self.entities:
                    if ne1["text"] != ne2["text"]:
                        arg1 = ne1
                        arg2 = ne2

                        relevantData = sentenceData  # sentenceData[arg1I:arg2I + 1]

                        isAdded = False
                        if id in self.annotations:
                            for LINK, ARG1, ARG2 in self.annotations[id]:
                                if arg1["text"] == ARG1 and arg2["text"] == ARG2:
                                    if LINK in TAGS:
                                        data.append(((arg1, arg2, relevantData), LINK))
                                        isAdded = True
                                        isAnyAdded = True
                        if not isAdded:
                            data.append(((arg1, arg2, relevantData), NO_CONNECTION))

                            # if not isAnyAdded and len(self.annotations[id]) > 0:
                            #     print "--------"
                            #     print self.annotations[id]
                            #     for e in entities:
                            #         print e
                            #     print parsed.ents
                            #     print "--------"
        return data

    def readAnnotations(self):
        """
        reads annotations
        :return: annotations
        """
        if not self.trainAnnotations:
            return {}

        annotationsData = set(filter(lambda l: l != "", open(self.trainAnnotations).read().split("\n")))

        annotations = {}
        for annotation in annotationsData:
            ID, ARG1, LINK, ARG2, OTHER = annotation.split("\t")
            ARG1 = self.clean(ARG1)
            ARG2 = self.clean(ARG2)
            if ID not in annotations:
                annotations[ID] = []
            if LINK in TAGS:
                annotations[ID].append((LINK, ARG1, ARG2))
        return annotations

    def readCorpus(self):
        sentences = {}
        for line in codecs.open(self.trainFile, encoding="utf8"):
            sentId, sent = line.strip().split("\t")
            sent = sent.replace("-LRB-", "(").replace("-RRB-", ")")
            sentences[sentId] = sent
        return sentences

    def extract_output_lines(self, predicted):
        """
        used to print pred annotations
        :param predicted: predicted output from train
        :return: lines
        """
        lines = []
        for i, prediction in enumerate(predicted):
            if prediction == NO_CONNECTION:
                continue
            arg1, arg2, _ = self.data[i][0]
            lines.append('%s\t%s\t%s\t%s\t' % (arg1['id'], arg1['text'], prediction, arg2['text']))
        return lines

    def output_to_file(self,output_file_name, predicted):
        """
        writes pred annotations to file
        :param output_file_name: path to output file
        :param predicted: predicted output from train
        :return: N/A
        """
        with open(output_file_name, 'w') as output_file:
            output_file.write('\n'.join(self.extract_output_lines(predicted)))

def main((trainCorpusFile, trainAnnotationsFile)):
    if trainCorpusFile[:1] is '#':
        print 'use .txt files as an input instead of .processed'

    trainData = Data(trainCorpusFile, trainAnnotationsFile)

    model = Model()
    model.train(trainData.data)

    print "Total features:", len(model.features)

    pickle.dump(model, open("model", "w"))


if __name__ == "__main__":
    main(sys.argv[1:])