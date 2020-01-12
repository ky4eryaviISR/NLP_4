from data.spc import sentence_dict
from sklearn import svm

tag_dict = {'OrgBased_In':
                {
                    'before': [],
                    'after': []
                },
                'Located_In':
                    {
                        'before': [],
                        'after': []
                    },
                'Live_In':
                    {
                        'before': [],
                        'after': []
                     },
                'Work_For':
                    {
                        'before': [],
                        'after': [],
                    },
                'Kill':
                    {
                        'before': [],
                        'after': []
                    }
}

def main():
    print(sentence_dict)
    with open('data/TRAIN.annotations') as fp:
        for line in fp:
            sen_id = line.split()[0]
            words = line.split()[1:]
            index = words.index('(')
            words = words[:index]
            temp = sentence_dict[sen_id]
            splitter = [i for i in tag_dict.keys() if i in words][0]
            index = words.index(splitter)
            for i in range(index):
                x = [j for j in temp if j['text'] == words[i]]
                tag_dict[splitter]['before'].append(x)
            for i in range(index+1, len(words)):
                x = [j for j in temp if j['text'] == words[i]]
                tag_dict[splitter]['after'].append(x)

    dict_att = {}
    for key in tag_dict.keys():
        print('-----------------------',key,'------------------------')
        dict_att[key] = {}
        for att, word_list in tag_dict[key].items():
            print('-----------------------', att, '------------------------')
            dict_att[key][att] = {}
            for word in word_list:
                word = word[0]
                if word['ent_type'] not in dict_att[key][att]:
                    dict_att[key][att][word['ent_type']] = 1
                else:
                    dict_att[key][att][word['ent_type']] += 1
                print(word)


    model = svm.SVC()

    model.fit()






if __name__ == '__main__':
    main()