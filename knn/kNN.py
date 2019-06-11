from numpy import *
import csv


#读取
with open('attributes.csv', 'r') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]

#分组
random.shuffle(datas)      #打乱顺序
n = len(datas)//10   #此处//保证整除

test_set = datas[0:n]       #测试集
train_set = datas[n:]       #训练集

#KNN
#求距离（欧式）
def distance(d1, d2):
    res = 0
    for key in ("totalWordsCount", "semanticobjscore", "semanticsubjscore",	"CC", "CD",	"DT", "EX",	"FW", "INs", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT",	"POS", "PRP", "PRP$", "RB",	"RBR", "RBS", "RP",	"SYM", "TOs", "UH",	"VB", "VBD", "VBG",	"VBN", "VBP", "VBZ", "WDT",	"WP", "WP$", "WRB",	"baseform",	"Quotes", "questionmarks", "exclamationmarks", "fullstops",	"commas", "semicolon", "colon",  "ellipsis", "pronouns1st",	"pronouns2nd", "pronouns3rd", "compsupadjadv", "past", "imperative", "present3rd", "present1st2nd", "sentence1st", "sentencelast", "txtcomplexity"):
        res+=(float(d1[key])-float(d2[key]))**2

    return res**0.5

K=10
def knn (data):
    #1.距离
    res=[
        {"result": train['Label'], "distance":distance(data, train)}
        for train in train_set
        ]

    #2.排序——升序
    res = sorted(res, key = lambda item:item['distance'])

    #3.取前K个
    res2 = res[0:K]

    #4.加权平均
    result = {'subjective':0, 'objective': 0 }

    #总距离
    sum=0
    for r in res2:
        sum += r['distance']

    for r in res2:
        result[r['result']]+=1-r['distance']/sum

    print(result)
    print(data['Label'])

    #判定结果
    if result['subjective']>result['objective']:
        return 'subjective'
    else:
        return "objective"

knn(test_set[0])

#测试阶段
correct = 0
for test in test_set:
    result = test['Label']
    result2 = knn(test)

    if result == result2:
        correct += 1

print("准确率：{:.2f}%".format(100*correct/len(test_set)))