# -*- coding:utf-8 -*-
"""
作者:郑帅
日期:2022年10月20日
时间:09:56
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # 交叉验证法的库
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
resource_path = r"NOx_data.xlsx"
model_save_path = "E:\\code\\DATA\\NOx_Data\\NOx_model_1.h5"
# train_history_path = "E:\\code\\DATA\\NOx_Data\\NOx_train_process_1.txt"
train_history_path = r"train.xlsx"

# 1 训练
# 2 测试+输出
case = 1
def Data_prepared():
    # 数据有10000*30，其中【0】是顺序，【1】--【27】是数据，【28】【29】是NOX的输出值
    data = pd.read_excel(resource_path, sheet_name="original")
    data=np.array(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaler = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test =train_test_split(data_scaler[0:10000,0:28], data_scaler[0:10000,29], test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(data[0:10000, 0:28], data[0:10000, 29],
                                                        test_size=0.3, random_state=0)
    y_train=y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    return X_train, X_test, y_train, y_test,scaler
def Model():
    return Lasso()

if __name__ == "__main__":
    # X_train, X_test, y_train, y_test=[],[],[],[]
#121231241
    X_train, X_test, y_train, y_test,scaler=Data_prepared()
    alpha_range = np.logspace(-8, -2, 200, base=10)
    print(alpha_range)  # 200个自定义的alpha值

    # LassoCV
    model= LassoCV(alphas=alpha_range, cv=5,max_iter=10000)

    model.fit(X_train,y_train)

    best_alpha = model.alpha_
    each_five_alpha = model.mse_path_
    mean = model.mse_path_.mean(axis=1)  # 有注意到在岭回归中我们的轴向是axis=0吗?
    print(mean.shape)
    print(best_alpha)
    #print(each_five_alpha)
    w = model.coef_

    # 获取R2指数
    r2_score = model.score(X_test, y_test)  # 0.6038982670571436




    pred=model.predict(X_test)
    real=y_test
    # pred=lt.Inverse_Maxmin(pred,scaler)
    # real=lt.Inverse_Maxmin(y_test,scaler)

    MSE=0
    for i in range(0,pred.shape[0]-1):
        MSE+=(pow(abs(pred[i] - y_test[i]), 1))/y_test[i]
    MSE/=pred.shape[0]
    print(MSE)

    plt.figure(figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(real, color="orange", label="Real value")
    # plt.plot(pred, color="c", label="RNN predicted result")
    plt.plot(pred, color='r', label='LSTM predicted result')
    plt.legend()
    plt.xlabel("NOx_Emission")
    plt.ylabel("Time")


    plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    # plt.axis('square')
    plt.title('A_side_test')
    # 【150,550】切成100分
    n = np.linspace(150, 550, num=100)
    m = n
    plt.plot(n, m, color="orange", label="Perfect_line")
    plt.scatter(real, pred, color="blue", label="A_side_data")
    plt.xlabel("Measured_NOx_Emission")
    plt.ylabel("Estimated_NOx_Emission")
    plt.legend()

    plt.grid(True)
    plt.show()

