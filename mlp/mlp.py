import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score

from sklearn import linear_model

df = pd.read_csv('attributes.csv')

df.head()                                 #返回前五个数据 

df.columns                                  #数据框索引列表

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow import feature_column

tf.random.set_seed(1)                                                    #设定随机种子值保证运行结果不变

train, test = train_test_split(df, test_size=0.2, random_state=1)        #划分数据集为训练集和测试集，比例为80：20

train, valid = train_test_split(train, test_size=0.2, random_state=1)    #划分训练集集为最终的训练集和验证集，比例为80：20

#验证划分好的数据长度
print(len(train))
print(len(valid))
print(len(test))

feature_columns = []          #创建一个特征列表

#数字特征列表
numeric_columns = ['semanticobjscore', 'semanticsubjscore',	'CC', 'CD',	'DT', 'EX',	'FW', 'INs', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',	'POS', 'PRP', 'PRPS', 'RB',	'RBR', 'RBS', 'RP',	'SYM', 'TOs', 'UH',	'VB', 'VBD', 'VBG',	'VBN', 'VBP', 'VBZ', 'WDT',	'WP', 'WPS', 'WRB',	'baseform',	'Quotes', 'questionmarks', 'exclamationmarks', 'fullstops',	'commas', 'semicolon', 'colon',  'ellipsis', 'pronouns1st',	'pronouns2nd', 'pronouns3rd', 'compsupadjadv', 'past', 'imperative', 'present3rd', 'present1st2nd', 'sentence1st', 'sentencelast', 'txtcomplexity']

#将数字特征列表加入空的特征列表中
for header in numeric_columns:
  feature_columns.append(feature_column.numeric_column(header))

#查看特征列表
feature_columns

from tensorflow.keras import layers

#创建特征层Densefeatures,输入为特征列表
feature_layer = layers.DenseFeatures(feature_columns)

#该函数的功能是将pandas数据框转化为tensorflow数据流
def df_to_tfdata(df, shuffle=True, bs=32):
  df = df.copy()
  labels = df.pop('Label')
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(df), seed=1)
  ds = ds.batch(bs)
  return ds

#将分好的数据转化为数据流
train_ds = df_to_tfdata(train)
valid_ds = df_to_tfdata(valid, shuffle=False)
test_ds = df_to_tfdata(test, shuffle=False)

#使用keras建立贯序模型，特征层作为输入层，隐藏层有100个节点，使用relu激活函数，输出层为1个神经元，使用sigmoid激活函数
model = keras.Sequential([
  feature_layer,
  layers.Dense(100, activation='relu'),
  layers.Dense(100, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

#配置模型，使用adam优化器，二分类使用binary_crossentropy损失函数，评估参数为准确度
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#把训练集和验证集放入构建好的模型，训练5个轮次
model.fit(train_ds,
          validation_data=valid_ds,
          epochs=5)

#查看模型
model.summary()

#使用测试集输出f1_score
f1_score(test.Label, np.rint(model.predict(test_ds)), average='micro')

#打印测试集正确率
model.evaluate(test_ds)

#输出测试集的混淆矩阵
confusion_matrix(test.Label, np.rint(model.predict(test_ds)))