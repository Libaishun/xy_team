# --*-- coding: utf-8 --*--

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import time

# 训练数据
df = pd.read_csv("d_train_20180102.csv",encoding="gb2312")
df_columns=df.columns
# print df.mean()
# 测试数据
df_test = pd.read_csv("d_test_A_20180102.csv",encoding="gb2312")

# 血糖值大于6.1的为糖尿病
df_pos = df[df['血糖'.decode("utf-8")]>=6.1]
df_neg = df[df['血糖'.decode("utf-8")]<6.1]

print "前四列是：",df_columns[:4]

def process_input(df_,scaler=None):
    df0 = df_.fillna(df.mean())
    df1 = df_.fillna(-100.)
    x0 = df0.values[:, 4:41]
    x1 = df1.values[:, 4:41]
    age0 = df0["年龄".decode("utf-8")].values
    x0 = np.hstack((x0, age0.reshape((-1, 1))))
    age1 = df1["年龄".decode("utf-8")].values
    x1 = np.hstack((x1, age1.reshape((-1, 1))))
    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(x0)
    x0 = scaler.transform(x0)
    x_copy = np.zeros(x0.shape)
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            if x1[i, j] == -100.:
                x0[i, j] = 0.
                x_copy[i, j] = 1
    x = np.hstack((x0, x_copy))
    sex = df_["性别".decode("utf-8")].values
    sex_hot = []
    for s in sex:
        if "男".decode("utf-8") in s:
            sex_hot.append([1, 0])
        else:
            sex_hot.append([0, 1])
    sex_hot = np.array(sex_hot)
    x = np.hstack((x, sex_hot))
    print x.shape
    return x,scaler

# 特征处理
x_dtest, scaler = process_input(df_test,scaler=None)
x_dtrain_pos, _ = process_input(df_pos,scaler=scaler)
x_dtrain_neg, _ = process_input(df_neg,scaler=scaler)

def draw_compare(df1, df2, name1, name2):
    for cl_name in df1.columns:
        try:
            cl1 = df1[cl_name]
            cl2 = df2[cl_name]
            plt.figure(num=1, figsize=(8, 6))
            plt.hist(cl1, bins=40, range=(cl1.min(), cl1.max()), color='b', label=name1)
            plt.hist(cl2, bins=40, range=(cl2.min(), cl2.max()), color='r', label=name2)
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
            plt.savefig("features/" + cl_name + ".png")
            plt.close(1)
        except:
            pass

# 比较训练集和测试集的特征分布
draw_compare(df, df_test, 'train', 'test')

y = df['血糖'.decode("utf-8")].values
y_mean = y.mean()
print "y 的平均值为 %f"%y_mean

y_pos = df_pos['血糖'.decode("utf-8")].values
y_neg = df_neg['血糖'.decode("utf-8")].values

print x_dtrain_pos.shape, y_pos.shape, x_dtrain_neg.shape, y_neg.shape

import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
import keras
from keras.callbacks import TensorBoard,EarlyStopping

def run_model(x_dtrain,y,x_dtest):
    shuff_index = [i for i in range(x_dtrain.shape[0])]
    random.shuffle(shuff_index)
    cut=int(0.8*len(shuff_index))

    # 将数据集切割成 训练集 测试集
    x_train = np.array([x_dtrain[i] for i in shuff_index[:cut]])
    y_train = np.array([y[i] for i in shuff_index[:cut]])

    x_train = np.vstack((x_dtrain_pos, x_train))
    y_train = np.hstack((y_pos, y_train))

    x_test = np.array([x_dtrain[i] for i in shuff_index[cut:]])
    y_test = np.array([y[i] for i in shuff_index[cut:]])

    # 搭建模型
    model = Sequential()
    model.add(Dense(units=300, activation='relu', kernel_initializer='random_uniform', input_dim=x_dtrain.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=300, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001,decay=1e-6)
    model.compile(loss='mean_squared_error',
                  optimizer=adam)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01,
                           patience=100, verbose=0, mode='min')
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)
    # 训练模型
    model.fit(x_train,
              y_train,
              epochs=1000,
              batch_size=64,
              validation_split=0.2,
              shuffle=True,
              callbacks=[early_stop])

    y_pre = model.predict(x_test)
    y_pre = y_pre.reshape((y_pre.shape[0]))
    score = np.mean(np.square(y_pre-y_test))
    print y_pre.shape, y_test.shape, score

    y_upload = model.predict(x_dtest)
    y_upload = y_upload.reshape((y_upload.shape[0]))
    model = []

    return score, y_upload

scores = []
predictions = []
for i in range(10):
    score, y_upload = run_model(x_dtrain_neg,y_neg,x_dtest)
    scores.append(score)
    predictions.append(y_upload)
    f=open(time.strftime("results/upload/dark_master%Y%m%d_v"+str(i)+".csv",time.localtime(time.time())),"w")
    for v in y_upload:
        f.write(str(round(v,3))+"\n")
    f.close()

def draw_curves(y_mul,name):
    x_0 = np.arange(len(y_mul[0]))
    plt.figure(num=1, figsize=(10,5))
    for i,y_0 in enumerate(y_mul):
        plt.plot(x_0,y_0,label="v_"+str(i))
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig("results/"+name+".png")
    plt.close(1)

draw_curves([scores],"score")
draw_curves(predictions,"upload")
f=open("results/score","w")
for s in scores:
    f.write(str(round(s,3))+"\n")
f.close()

draw_curves([y],'xuetang')
