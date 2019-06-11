from __future__ import print_function
from numpy import *

from nltk.corpus import stopwords



def createVocabList(dataSet):

    vocabSet = set([])  # create empty set

    for document in dataSet:

        # 操作符 | 用于求两个集合的并集

        vocabSet = vocabSet | set(document)  # union of the two sets

    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):

    # 创建一个和词汇表等长的向量，并将其元素都设置为0

    returnVec = [0] * len(vocabList)# [0,0......]

    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1

    for word in inputSet:

        if word in vocabList:

            returnVec[vocabList.index(word)] = 1

        else:

            print("the word: %s is not in my Vocabulary!" % word)

    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    #文档词袋模型

    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):

    # 总文件数

    numTrainDocs = len(trainMatrix)

    # 总单词数

    numWords = len(trainMatrix[0])

    # 主观文本的出现概率

    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 构造单词出现次数列表

    # p0Num 客观的统计

    # p1Num 主观的统计

    # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1

    p0Num = ones(numWords)#[0,0......]->[1,1,1,1,1.....]

    p1Num = ones(numWords)




    p0Denom = 2.0

    p1Denom = 2.0

    for i in range(numTrainDocs):

        if trainCategory[i] == 1:

            # 累加主观词的频次

            p1Num += trainMatrix[i]

            # 对每篇文章的主观性词汇的频次 进行统计汇总

            p1Denom += sum(trainMatrix[i])

        else:

            p0Num += trainMatrix[i]

            p0Denom += sum(trainMatrix[i])

    # 类别1，即主观性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表

    p1Vect = log(p1Num / p1Denom)

    # 类别0，即客观文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表

    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive





def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):


    p1 = sum(vec2Classify * p1Vec) + log(pClass1)

    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:

        return 1

    else:

        return 0






def textParse(bigString):
    '''文本文件解析，返回字符串列表'''
    import re
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~]+'
    listOfTokens =str(bigString.lower())
    listOfTokens=re.sub(r,' ',listOfTokens)
    words=listOfTokens.split(' ')
    stop_words = set(stopwords.words('english'))
    filtered_words=[]
    for w in words:
        if  w not in stop_words:
            filtered_words.append(w)
    return(filtered_words)

def objectiveTest():

    '''

    Desc:

        对贝叶斯主观性文本分类器进行自动化处理。

    Args:

        none

    Returns:

        对测试集中的每个文本进行分类，若文档分类错误，则错误数加 1，最后返回总的错误百分比。

    '''

    docList = []

    classList = []

    fullText = []

    for i in range(1, 360):

        # 切分，解析数据，并归类为 1 类别

        wordList = textParse(open('D:/Users/Administrator/PycharmProjects/bayes/dataSet/objective/{}.txt'.format(i),encoding='gb18030',errors='ignore').read())

        docList.append(wordList)

        classList.append(1)

        # 切分，解析数据，并归类为 0 类别

        wordList = textParse(open('D:/Users/Administrator/PycharmProjects/bayes/dataSet/subjective/{}.txt'.format(i),encoding='gb18030',errors='ignore').read())

        docList.append(wordList)

        fullText.extend(wordList)

        classList.append(0)


    # 创建词汇表

    vocabList = createVocabList(docList)

    trainingSet = list(range(718))


    testSet = []

    # 随机取 100 个文本用来测试

    for i in range(100):

        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数

        randIndex = int(random.uniform(0, len(trainingSet)))

        testSet.append(trainingSet[randIndex])

        del(trainingSet[randIndex])

    trainMat = []

    trainClasses = []


    for docIndex in trainingSet:

        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))

        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    errorCount = 0

    for docIndex in testSet:

        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])

        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:

            errorCount += 1

    print('the errorCount is: ', errorCount)

    print('the testSet length is :', len(testSet))

    print('the error rate is :', float(errorCount)/len(testSet))


def testParseTest():

    print(textParse(open('D:/Users/Administrator/PycharmProjects/bayes/dataSet/objective/1.txt',encoding='gb18030',errors='ignore').read()))


objectiveTest()
