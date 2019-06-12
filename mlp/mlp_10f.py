import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score

from sklearn import linear_model

df = pd.read_csv('attributes.csv')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow import feature_column

feature_columns = []          #创建一个特征列表

#数字特征列表
numeric_columns = ['semanticobjscore', 'semanticsubjscore',	'CC', 'CD',	'DT', 'EX',	'FW', 'INs', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',	'POS', 'PRP', 'PRPS', 'RB',	'RBR', 'RBS', 'RP',	'SYM', 'TOs', 'UH',	'VB', 'VBD', 'VBG',	'VBN', 'VBP', 'VBZ', 'WDT',	'WP', 'WPS', 'WRB',	'baseform',	'Quotes', 'questionmarks', 'exclamationmarks', 'fullstops',	'commas', 'semicolon', 'colon',  'ellipsis', 'pronouns1st',	'pronouns2nd', 'pronouns3rd', 'compsupadjadv', 'past', 'imperative', 'present3rd', 'present1st2nd', 'sentence1st', 'sentencelast', 'txtcomplexity']

#将数字特征列表加入空的特征列表中
for header in numeric_columns:
  feature_columns.append(feature_column.numeric_column(header))

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

from sklearn.model_selection import KFold, StratifiedKFold

kf = KFold(n_splits=10, shuffle = True)

#下面做10折交叉验证

accuracy_10f = []           #创建一个空列表用来保存每折的准确率

for train_index,test_index in kf.split(df):
  train = df.ix[train_index]
  test = df.ix[test_index]
  train, valid = train_test_split(train, test_size=0.2, random_state=1)
  
  train_ds = df_to_tfdata(train)
  valid_ds = df_to_tfdata(valid, shuffle=False)
  test_ds = df_to_tfdata(test, shuffle=False)
  
  model.fit(train_ds,
            validation_data=valid_ds,
            epochs=5)
  
  accuracy = model.evaluate(test_ds)[1]
  accuracy = accuracy_10f.append(accuracy)
  f1 = f1_score(test.Label, np.rint(model.predict(test_ds)), average='micro')
  print(f1)
  con = confusion_matrix(test.Label, np.rint(model.predict(test_ds)))
  print(con)
  print('next_f')

print(accuracy_10f)

#求平均值
sum = 0
for i in range(10):
  sum += accuracy_10f[i]
average = sum/10
print('The average accuracy is',average)