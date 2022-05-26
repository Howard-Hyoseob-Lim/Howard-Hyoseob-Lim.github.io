---
layout : single
title : "Predict who will purchase next week by Deep Learning"
categories : Deep_Learning
tag : [python, Deep_Learning, business, GA]
toc : true
publised : true
author_profile : false
---


```python
import os
import pandas as pd
import numpy as np
```


```python
#Data Preprocessing
result2 = result2.dropna()
#columns rename 
result2.rename(columns = {'2. 고객 ID':'customer_id'}, inplace = True)
result2.rename(columns = {'5. 고객등급':'grade'}, inplace = True)
result2.rename(columns = {'날짜':'date'}, inplace = True)
result2.rename(columns = {'세션 수':'session'}, inplace = True)
result2.rename(columns = {'순 이벤트 수':'event'}, inplace = True)
result2.rename(columns = {'순 페이지뷰 수':'pageview'}, inplace = True)
result2.rename(columns = {'세션 시간':'session_time'}, inplace = True)
result2.rename(columns = {'평균 페이지에 머문 시간':'page_time'}, inplace = True)
result2.rename(columns = {'거래수':'quantity'}, inplace = True)
result2.rename(columns = {'상품 수익':'revenue'}, inplace = True)
#to_datetime
result2["date"] = pd.to_datetime(result2['date'])
```


```python
#Make target column
result2["target"] =  np.where(result2["quantity"]>0, 1, 0)
```


```python
#220515 Family+로 추출하여 테스트 시도
result3 = result2[result2["grade"]=="Family+"]
```


```python
df_feat = result3.drop("target", axis = 1)
df_target = result3[["target"]]
customer_list = result3["customer_id"].unique()
date_range = result3["date"].unique()
```


```python
customer_list
```




    array(['MB202010070066006', 'MBR000000464681', 'MBR000000394781', ...,
           'MB202103270198595', 'MBR000000916188', 'MB202108270376286'],
          dtype=object)




```python
mask = (result3["date"]>="2021-02-01") & (result3["date"]<="2021-02-07")
```


```python
filtered_result = result3.loc[mask]
```


```python
#학습 Target Label 구성 : 일주일 후 구매/비구매 여부
target_label_table = filtered_result.groupby('customer_id')
target_label_table2 = target_label_table.sum()
target_label_column = target_label_table2["target"]
```


```python
target_label = pd.merge(family_customer, target_label_column, how='left',on=['customer_id'])
```


```python
for i in target_label[(target_label['target']>0)].index:
    target_label.at[i,'target'] = 1
```


```python
target_label.fillna(0.0, inplace=True)
```


```python
target_label = target_label.set_index("customer_id")
```


```python
#1월 한 달 간의 데이터 생성
date_range = date_range[:30]
pro = pd.MultiIndex.from_product([customer_list,date_range], names=["customer_id","date"])
preprocessing = pd.DataFrame(index=pro).reset_index()
```


```python
#Feature 데이터와 테이블 결합하기 + Panel Data 생성하기
merge1 = pd.merge(preprocessing, result3, how='left',on=['customer_id', 'date'])
merge1["session"].fillna(int(merge1['session'].min()), inplace=True)
merge1["event"].fillna(int(merge1['event'].min()), inplace=True)
merge1["pageview"].fillna(int(merge1['pageview'].min()), inplace=True)
merge1["target"].fillna(0.0, inplace=True)
merge2 = merge1.drop(columns=["grade","session_time","page_time","quantity","revenue"])
```


```python
merge2 = merge2.set_index(["customer_id","date"])
```


```python
merge2.sort_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>session</th>
      <th>event</th>
      <th>pageview</th>
      <th>target</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">MB202004290002256</th>
      <th>2021-01-01</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-02</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-03</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-04</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">MBR000001308166</th>
      <th>2021-01-26</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-27</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-28</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-29</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-30</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1058550 rows × 4 columns</p>
</div>




```python
#Panel Data 학습 데이터화 (백터화)
res = merge2.to_xarray()
image = merge2.to_xarray().to_array().to_numpy()
re_image = image.reshape(image.shape[1],30,4)
```


```python
#훈련데이터/테스트데이터 분리
train_split = 20000
train_data = re_image[:train_split]
test_data = re_image[train_split:]
target = target_label["target"].to_numpy()
train_label = target[:train_split]
test_label = target[train_split:]
```

# SMOTE 알고리즘으로 데이터 불균형 해결


```python
pip install scipy
```

    Requirement already satisfied: scipy in c:\users\msi\anaconda3\envs\tfstart\lib\site-packages (1.5.2)
    Requirement already satisfied: numpy>=1.14.5 in c:\users\msi\anaconda3\envs\tfstart\lib\site-packages (from scipy) (1.19.2)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: You are using pip version 21.3.1; however, version 22.1 is available.
    You should consider upgrading via the 'C:\Users\MSI\Anaconda3\envs\tfstart\python.exe -m pip install --upgrade pip' command.
    


```python
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
```


```python
sm = SMOTE(sampling_strategy=0.5, random_state=10)
```


```python
dim_1 = np.array(train_data).shape[0]
dim_2 = np.array(train_data).shape[1]
dim_3 = np.array(train_data).shape[2]
```


```python
new_dim = dim_1 * dim_2
```


```python
new_x_train = np.array(train_data).reshape(new_dim, dim_3)
```


```python
new_y_train = []
for i in range(len(train_label)):
    # print(y_train[i])
    new_y_train.extend([train_label[i]]*dim_2)
```


```python
new_y_train = np.array(new_y_train)
```


```python
oversample = SMOTE()
X_Train, Y_Train = oversample.fit_resample(new_x_train, new_y_train)
# summarize the new class distribution
counter = Counter(Y_Train)
print('The number of samples in TRAIN: ', counter)
```

    The number of samples in TRAIN:  Counter({1.0: 577290, 0.0: 577290})
    


```python
x_train_SMOTE = X_Train.reshape(int(X_Train.shape[0]/dim_2), dim_2, dim_3)

y_train_SMOTE = []
for i in range(int(X_Train.shape[0]/dim_2)):
    # print(i)
    value_list = list(Y_Train.reshape(int(X_Train.shape[0]/dim_2), dim_2)[i])
    # print(list(set(value_list)))
    y_train_SMOTE.extend(list(set(value_list)))
    ## Check: if there is any different value in a list 
    if len(set(value_list)) != 1:
        print('\n\n********* STOP: THERE IS SOMETHING WRONG IN TRAIN ******\n\n')
```


```python
y_train_SMOTE = np.array(y_train_SMOTE)
```

# CNN 으로 학습하기


```python
# CNN
import tensorflow as tf
import tensorflow_addons as tfa
```

    C:\Users\MSI\Anaconda3\envs\tfstart\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.7.0 and strictly below 2.10.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.3.0 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    


```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, padding="same",activation='relu', input_shape=(30,4)),
    tf.keras.layers.MaxPooling1D( 2 ),
    tf.keras.layers.Conv1D(64, 3, padding="same",activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128, 3, padding="same",activation='relu'),
    tf.keras.layers.MaxPooling1D( 2 ),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d (Conv1D)              (None, 30, 32)            416       
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 15, 32)            0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 15, 64)            6208      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 7, 64)             0         
    _________________________________________________________________
    dropout (Dropout)            (None, 7, 64)             0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 7, 128)            24704     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 3, 128)            0         
    _________________________________________________________________
    dense (Dense)                (None, 3, 128)            16512     
    _________________________________________________________________
    flatten (Flatten)            (None, 384)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 385       
    =================================================================
    Total params: 48,225
    Trainable params: 48,225
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.Accuracy(),
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall(),
                                                                     tfa.metrics.F1Score(num_classes=1,
                                                                                        average='macro',
                                                                                        threshold=0.5)])
```


```python
model.fit(x_train_SMOTE, y_train_SMOTE, epochs=10)
```

    Epoch 1/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3627 - accuracy: 0.1172 - precision_2: 0.8500 - recall_2: 0.7200 - f1_score: 0.7796: 1s - loss: 0.3654 - accuracy: 0.1173 - precision
    Epoch 2/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3624 - accuracy: 0.1177 - precision_2: 0.8606 - recall_2: 0.7092 - f1_score: 0.7776
    Epoch 3/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3593 - accuracy: 0.1207 - precision_2: 0.8693 - recall_2: 0.7042 - f1_score: 0.7780
    Epoch 4/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3590 - accuracy: 0.1230 - precision_2: 0.8607 - recall_2: 0.7122 - f1_score: 0.7794
    Epoch 5/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3580 - accuracy: 0.1206 - precision_2: 0.8577 - recall_2: 0.7177 - f1_score: 0.7815
    Epoch 6/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3560 - accuracy: 0.1201 - precision_2: 0.8780 - recall_2: 0.6980 - f1_score: 0.7777
    Epoch 7/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3550 - accuracy: 0.1166 - precision_2: 0.8683 - recall_2: 0.7055 - f1_score: 0.7785
    Epoch 8/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3542 - accuracy: 0.1178 - precision_2: 0.8711 - recall_2: 0.7081 - f1_score: 0.7812
    Epoch 9/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3530 - accuracy: 0.1247 - precision_2: 0.8691 - recall_2: 0.7093 - f1_score: 0.7811
    Epoch 10/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3520 - accuracy: 0.1250 - precision_2: 0.8801 - recall_2: 0.7002 - f1_score: 0.7799
    




    <tensorflow.python.keras.callbacks.History at 0x2aa9ed88ee0>




```python
test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
```

    478/478 - 0s - loss: 0.1671 - accuracy: 0.1576 - precision_2: 0.0000e+00 - recall_2: 0.0000e+00 - f1_score: 0.0000e+00
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-59-8b4395f9a420> in <module>
    ----> 1 test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
    

    ValueError: too many values to unpack (expected 2)


# VGG 모델 적용


```python
model = tf.keras.Sequential([
  tf.keras.layers.Conv1D(input_shape=(30,4), kernel_size=(3), filters=32, padding='same', activation='relu'), 
  tf.keras.layers.Conv1D(kernel_size=(3), filters=64, padding='same', activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=(2)), 
  tf.keras.layers.Dropout(rate=0.5), 
  tf.keras.layers.Conv1D(kernel_size=(3), filters=128, padding='same', activation='relu'),    
  tf.keras.layers.Conv1D(kernel_size=(3), filters=256, padding='valid', activation='relu'),  
  tf.keras.layers.MaxPool1D(pool_size=(2)),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(units=512, activation='relu'), 
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(units=256, activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(units=1, activation='sigmoid')                           
])

```


```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.Accuracy(),
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall(),
                                                                     tfa.metrics.F1Score(num_classes=1,
                                                                                        average='macro',
                                                                                        threshold=0.5)])
```


```python
model.fit(x_train_SMOTE, y_train_SMOTE, epochs=10)
```

    Epoch 1/10
    1203/1203 [==============================] - 11s 10ms/step - loss: 0.4685 - accuracy: 0.0176 - precision_3: 0.6813 - recall_3: 0.9111 - f1_score: 0.7796
    Epoch 2/10
    1203/1203 [==============================] - 11s 9ms/step - loss: 0.4001 - accuracy: 0.0333 - precision_3: 0.8060 - recall_3: 0.7320 - f1_score: 0.7672
    Epoch 3/10
    1203/1203 [==============================] - 11s 9ms/step - loss: 0.3891 - accuracy: 0.0293 - precision_3: 0.8429 - recall_3: 0.7062 - f1_score: 0.7686
    Epoch 4/10
    1203/1203 [==============================] - 11s 9ms/step - loss: 0.3855 - accuracy: 0.0385 - precision_3: 0.8507 - recall_3: 0.7036 - f1_score: 0.7702
    Epoch 5/10
    1203/1203 [==============================] - 11s 9ms/step - loss: 0.3849 - accuracy: 0.0501 - precision_3: 0.8568 - recall_3: 0.6961 - f1_score: 0.7681
    Epoch 6/10
    1203/1203 [==============================] - 12s 10ms/step - loss: 0.3836 - accuracy: 0.1836 - precision_3: 0.8603 - recall_3: 0.6955 - f1_score: 0.7691
    Epoch 7/10
    1203/1203 [==============================] - 11s 9ms/step - loss: 0.3814 - accuracy: 0.1957 - precision_3: 0.8637 - recall_3: 0.6964 - f1_score: 0.7711
    Epoch 8/10
    1203/1203 [==============================] - 11s 10ms/step - loss: 0.3813 - accuracy: 0.1265 - precision_3: 0.8621 - recall_3: 0.6960 - f1_score: 0.7702
    Epoch 9/10
    1203/1203 [==============================] - 12s 10ms/step - loss: 0.3789 - accuracy: 0.1084 - precision_3: 0.8643 - recall_3: 0.6912 - f1_score: 0.7681
    Epoch 10/10
    1203/1203 [==============================] - 12s 10ms/step - loss: 0.3777 - accuracy: 0.0721 - precision_3: 0.8698 - recall_3: 0.6925 - f1_score: 0.7711
    




    <tensorflow.python.keras.callbacks.History at 0x2aa9fb87f40>



# Vanila LSTM 적용


```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(30,4), return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```


```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.Accuracy(),
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall(),
                                                                     tfa.metrics.F1Score(num_classes=1,
                                                                                        average='macro',
                                                                                        threshold=0.5)])
```


```python
model.fit(x_train_SMOTE, y_train_SMOTE, epochs=10)
```

    Epoch 1/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4948 - accuracy: 0.0000e+00 - precision_4: 0.6536 - recall_4: 0.9990 - f1_score: 0.7902
    Epoch 2/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4960 - accuracy: 0.0000e+00 - precision_4: 0.6540 - recall_4: 0.9986 - f1_score: 0.7904
    Epoch 3/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4952 - accuracy: 0.0000e+00 - precision_4: 0.6547 - recall_4: 0.9992 - f1_score: 0.7911
    Epoch 4/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4911 - accuracy: 0.0000e+00 - precision_4: 0.6567 - recall_4: 0.9993 - f1_score: 0.7925
    Epoch 5/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4970 - accuracy: 0.0000e+00 - precision_4: 0.6584 - recall_4: 0.9892 - f1_score: 0.79062s - loss: 0.4978 
    Epoch 6/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4847 - accuracy: 0.0000e+00 - precision_4: 0.6637 - recall_4: 0.9944 - f1_score: 0.7960
    Epoch 7/10
    1203/1203 [==============================] - 19s 16ms/step - loss: 0.4739 - accuracy: 0.0000e+00 - precision_4: 0.6674 - recall_4: 0.9804 - f1_score: 0.7942
    Epoch 8/10
    1203/1203 [==============================] - 18s 15ms/step - loss: 0.4523 - accuracy: 0.0000e+00 - precision_4: 0.6877 - recall_4: 0.9093 - f1_score: 0.78321s - loss: 0.4497 - accuracy: 0.00
    Epoch 9/10
    1203/1203 [==============================] - 19s 15ms/step - loss: 0.4357 - accuracy: 0.0000e+00 - precision_4: 0.7075 - recall_4: 0.8947 - f1_score: 0.7902
    Epoch 10/10
    1203/1203 [==============================] - 19s 16ms/step - loss: 0.4056 - accuracy: 0.0000e+00 - precision_4: 0.7614 - recall_4: 0.7974 - f1_score: 0.7790
    




    <tensorflow.python.keras.callbacks.History at 0x2aab0e20100>



# 제안 모델 CNN + LSTM


```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(input_shape=(30,4), kernel_size=3, filters=32, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(16, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```


```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.Accuracy(),
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall(),
                                                                     tfa.metrics.F1Score(num_classes=1,
                                                                                        average='macro',
                                                                                        threshold=0.5)])
```


```python
model.fit(x_train_SMOTE, y_train_SMOTE, epochs=10)
```

    Epoch 1/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.4750 - accuracy: 0.0000e+00 - precision_6: 0.6762 - recall_6: 0.9265 - f1_score: 0.7818
    Epoch 2/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.4364 - accuracy: 0.0000e+00 - precision_6: 0.7113 - recall_6: 0.8487 - f1_score: 0.7739
    Epoch 3/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.4219 - accuracy: 0.0000e+00 - precision_6: 0.7364 - recall_6: 0.8102 - f1_score: 0.7715
    Epoch 4/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.4126 - accuracy: 0.0000e+00 - precision_6: 0.7512 - recall_6: 0.7927 - f1_score: 0.7714
    Epoch 5/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.4055 - accuracy: 0.0000e+00 - precision_6: 0.7639 - recall_6: 0.7767 - f1_score: 0.7703
    Epoch 6/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.4038 - accuracy: 0.0000e+00 - precision_6: 0.7770 - recall_6: 0.7612 - f1_score: 0.7690
    Epoch 7/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3986 - accuracy: 0.0000e+00 - precision_6: 0.7887 - recall_6: 0.7475 - f1_score: 0.7675
    Epoch 8/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3959 - accuracy: 0.0000e+00 - precision_6: 0.7879 - recall_6: 0.7489 - f1_score: 0.7679
    Epoch 9/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3923 - accuracy: 0.0000e+00 - precision_6: 0.7945 - recall_6: 0.7443 - f1_score: 0.7686
    Epoch 10/10
    1203/1203 [==============================] - 4s 3ms/step - loss: 0.3922 - accuracy: 0.0000e+00 - precision_6: 0.7979 - recall_6: 0.7440 - f1_score: 0.7700
    




    <tensorflow.python.keras.callbacks.History at 0x2aab2e47df0>


