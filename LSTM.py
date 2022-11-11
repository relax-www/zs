# -*- coding:utf-8 -*-
"""
作者:郑帅
日期:2022年09月24日
时间:19:30
"""
import os
import numpy as np
import tensorflow
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
# import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # 交叉验证法的库
import openpyxl as xl
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

resource_path = r"NOx_data.xlsx"
model_save_path = r"NOx_model_1.h5"
# train_history_path = "E:\\code\\DATA\\NOx_Data\\NOx_train_process_1.txt"
train_history_path = r"train.xlsx"
LSTM_Model_Png = r'model.png'
Time_Step = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Learn_Rate = [0.1, 0.01, 0.001, 0.0001]
Hidden_Node = [16, 32, 64, 128, 256]

n_split = 5
time_step = 10
learn_rate = 0.01
hidden_node = 256
# 1 训练
# 2 测试+输出
case = 2


# root_mean_squared_error是MSE的开方
# mean_squared_error 是MSE的keras函数
def RMSE(y_true, y_pred):
    return pow(abs(keras.losses.mean_squared_error(y_true, y_pred)), 0.5)


def LSTM(x_train, y_train):
    avg_accuracy = 0
    avg_loss = 0

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=hidden_node, activation='tanh', recurrent_activation="sigmoid"
                                ,kernel_regularizer=keras.regularizers.l1(0.001),input_shape=(time_step, 30)))
    # model.add(keras.layers.LSTM())input_dim=30,input_length=4,
    # ten units LSTM
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0
                                 , amsgrad=False)

    # 文中使用RMSE和MAPE
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['Accuracy', 'mae'])

    # 过拟合常用正则和早停，文中使用早停=30，其余参数未知
    # verbose = 0 不在标准输出流输出日志信息
    # verbose = 1 输出进度条记录，进度条如 [====>…] - ETA
    # verbose = 2 每个epoch输出一行记录
    Early_Stop = keras.callbacks.EarlyStopping(monitor='Accuracy', patience=20, verbose=2, mode='auto')

    # train_x=[],train_y=[],test_x=[],test_y=[]
    kfold = KFold(n_splits=5, shuffle=False)  # 由于是LSTM的代码所以不需要random_state
    i=0
    print(time_step,learn_rate,hidden_node,sep=' ')
    for train_index, test_index in kfold.split(x_train, y_train):

        i+=1
        print("Time_Step")
        print("5折交叉验证法   第%d"%i+"次")
        print("test_index", test_index)
        train_x, test_x = x_train[train_index], x_train[test_index]
        train_y, test_y = y_train[train_index], y_train[test_index]

        # 此处  batch_size=16  为自己设置，没有依据，瞎蒙了一个   最后batch_size=1太慢了，还是16

        # hist = model.fit(train_x, train_y, batch_size=64, epochs=300, callbacks=[Early_Stop])
        hist = model.fit(train_x, train_y, batch_size=64, epochs=60,verbose=0)
        ##################记录训练的原始参数######################
        # frame_data = pd.DataFrame(hist.history)
        # frame_title = pd.DataFrame({'parameter': [time_step, learn_rate, hidden_node,i]})
        # with pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl', if_sheet_exists="overlay") as writer:
        #     Start_Row = 0
        #     Start_Col=0
        #     if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
        #         excel_data = pd.read_excel(train_history_path, sheet_name="Original_Train")
        #         Start_Col = excel_data.shape[1]
        #     else:
        #         Start_Col = 0
        #
        #     if Start_Col-4>=0:
        #         if excel_data.iloc[0, Start_Col - 4] == time_step \
        #                 and excel_data.iloc[1, Start_Col - 4] == learn_rate \
        #                 and excel_data.iloc[2, Start_Col - 4] == hidden_node:  # 反斜杠处不可以注释，同时反斜杠可以连接语句
        #             Start_Col = Start_Col - 4
        #             Start_Row = excel_data.iloc[:,Start_Col+2].shape[0]
        #
        #     frame_title.to_excel(writer, sheet_name="Original_Train"
        #                          , encoding='utf-8', startcol=Start_Col, startrow=Start_Row+1, index=False)
        #     frame_data.to_excel(writer, sheet_name="Original_Train"
        #                         , encoding='utf-8', startcol=Start_Col + 1, startrow=Start_Row+2, index=False)
        #     writer.save()
        ###################################            5折验证法          ##########################3
        print('Model evaluation: ', model.evaluate(test_x, test_y))
        avg_accuracy += model.evaluate(test_x, test_y)[1]
        avg_loss += model.evaluate(test_x, test_y)[0]

    print("K fold average accuracy: {}".format(avg_accuracy / n_split))
    print("K fold average loss: {}".format(avg_loss / n_split))
    frame_train_acc = pd.DataFrame({'1':[time_step],'2':[learn_rate],'3':[hidden_node]
                                       ,'acc':[avg_accuracy / n_split], 'loss':[avg_loss / n_split]})
    with pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl', if_sheet_exists="overlay") as writer:
        if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
            excel_data = pd.read_excel(train_history_path, sheet_name="train")
            Start_Col = excel_data.shape[1]
            Start_Row = excel_data.shape[0]
        else:
            Start_Col = 0
            Start_Row = 0
        frame_train_acc.to_excel(writer, sheet_name="train", encoding='utf-8', startcol=0, startrow=Start_Row+1,
                                 index=False,header=False)

    ########################################  5折交叉验证 ###############################################
    # scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')  # 返回值是  ndarray
    #
    #
    # Train_Acc = scores.mean()
    # Train_Confidence_Interval = scores.std() * 2
    # frame_train_acc=pd.DataFrame([Train_Acc,Train_Confidence_Interval])
    # print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    # with pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl', if_sheet_exists="overlay") as writer:
    #     if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
    #         excel_data = pd.read_excel(train_history_path, sheet_name="train")
    #         Start_Col = excel_data.shape[1]
    #         Start_Row = excel_data.shape[0]
    #     else:
    #         Start_Col = 0
    #         Start_Row = 0
    #     frame_train_acc.to_excel(writer, sheet_name="train", encoding='utf-8', startcol=0,startrow=Start_Row, index=False)
    #######################################################################################
    return model


def visualization(real, pred):
    plt.figure(figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(real, color="orange", label="Real value")
    # plt.plot(pred, color="c", label="RNN predicted result")
    plt.plot(pred, color='r', label='LSTM predicted result')
    plt.legend()
    plt.xlabel("NOx_Emission")
    plt.ylabel("Time")

    plt.figure(figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(real[7000:, 0], color="blue", label="Real value")
    # plt.plot(pred, color="c", label="RNN predicted result")
    plt.plot(pred[7000:, 0], color='green', label='LSTM predicted result')
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
    plt.scatter(real[7000:, 0], pred[7000:, 0], color="blue", label="A_side_data")
    plt.xlabel("Measured_NOx_Emission")
    plt.ylabel("Estimated_NOx_Emission")
    plt.legend()

    plt.grid(True)
    plt.show()


def Inverse_Maxmin(array, scaler):
    array=array.reshape(array.shape[0],1)
    array = np.tile(array, reps=(1, 30))
    array = scaler.inverse_transform(array)

    return array[:, 28]


# 函数用于输出预测值和真实值
def To_Excel_Result(real, pred):
    # replace overlay
    writer = pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl', if_sheet_exists="overlay")
    # ws=writer.create_sheet("Aside_result")
    # out_data=real.append()
    # 覆盖性写法
    out = np.concatenate((real, pred), axis=1)
    out = pd.DataFrame(out)
    out_title = pd.DataFrame(pd.DataFrame({'parameter': [time_step, learn_rate, hidden_node]}))
    # 主要是不想覆盖读取
    if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
        excel_data = pd.read_excel(train_history_path, sheet_name="Original_result")
        Start_Col = excel_data.shape[1]
    else:
        Start_Col = 0
    # print(excel_data.shape[1])
    # startcol=4 实际上是0-3  4开始
    out_title.to_excel(writer, sheet_name='Original_result', encoding='utf-8', startcol=Start_Col,
                       index=False)
    out.to_excel(writer, sheet_name='Original_result', encoding='utf-8', header=['real', 'pred'],
                 startcol=Start_Col + 1, index=False)
    writer.save()  # 保存后才能成功改动
    writer.close()


def To_Model_Png(model):
    keras.utils.plot_model(model, to_file=LSTM_Model_Png, show_shapes=True)


# n_steps=4，layers=1，hidden nodes=256, epochs=300, earlystop=30
if __name__ == "__main__":
    time_start = time.time()


    # 数据有10000*30，其中【0】是顺序，【1】--【27】是数据，【28】【29】是NOX的输出值
    data = pd.read_excel(resource_path, sheet_name="original")
    # 前7000组数据取做训练集，后三千组数据做测试集
    train_data = data.loc[:7000 - 1]
    # 处理完成的数组是ndarray类型，整合+归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaler = scaler.fit_transform(train_data)
    # A出口和B出口
    train_scaler_A = train_scaler[:, 28]
    train_scaler_B = train_scaler[:, 29]
    # print(train_scaler_A)

    if case == 1:
        X_train_A = []
        Y_train_A = []

        # LSTM的输入必须是  n组训练集中有时间index+属性columns 3维数组 ndarray
        for i in range(time_step, train_data.shape[0]):
            X_train_A.append(train_scaler[i - time_step:i, :])
            Y_train_A.append(train_scaler_A[i])
        # 此处的append之后带有array（），需要用np。array重新定义成一个数组
        X_train, Y_train = np.array(X_train_A), np.array(Y_train_A)
        # 定义成一个数组后，还需要调整一下形状
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        # print(X_train)

        ####################################  测试集处理  #####################################
        test_data = data.loc[7000 - time_step + 1:]
        test_data = scaler.transform(test_data)
        X_test = []
        Y_test = []
        for i in range(time_step, test_data.shape[0]):
            X_test.append(test_data[i - time_step:i, :])
            Y_test.append(test_data[i - time_step:i, 28])
        # 取29列就是A组出风口的NOx的输出值
        X_test = np.array(X_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
        X_test = np.append(X_train, X_test, 0)

        #####################################模型训练+打开####################################

        if case == 1:
            # 训练模型
            model = LSTM(X_train, Y_train)
            # learn_rate=0.001
            # model = LSTM(X_train, Y_train)
            # learn_rate = 0.01
            # model = LSTM(X_train, Y_train)
            # learn_rate = 0.1
            # model = LSTM(X_train, Y_train)
            # 保存现有模型
            model.save(model_save_path)
            # 删除当前已存在的模型
            # del model
        if case == 1:
            # 加载模型
            model = keras.models.load_model(model_save_path, compile=False, custom_objects={'RMSE': RMSE})
            #  model = load_model(model_save_path)
            # 测试集来预测相应的NOx输出
            pred = model.predict(X_test)
            # 对于pred进行反归一化，由于scaler的大小为[:,30],因此进行了一步转化；见函数
            pred = Inverse_Maxmin(pred, scaler)
            # pred=np.array(pred)
            pred = pred.reshape(pred.shape[0], 1)
            # 取出相应的测试集的真实输出
            real = data.loc[data.shape[0] - pred.shape[0]:, ["Unnamed: 28"]]
            real = np.array(real)
            # 可视化
            visualization(real=real, pred=pred)


            pred_acc = pd.DataFrame((abs(real - pred) / real))
            pred_acc_data=0
            for i in range(0,pred_acc.shape[0]-1):
                pred_acc_data+=pred_acc.iloc[i,0]
            pred_acc=pd.DataFrame({'1':[time_step],'2':[learn_rate],'3':[hidden_node],'acc':[pred_acc_data/pred_acc.shape[0]]})#行求和

            with pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl', if_sheet_exists="overlay") as writer:
                if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
                    excel_data = pd.read_excel(train_history_path, sheet_name="result")
                    Start_Col = excel_data.shape[1]
                    Start_Row = excel_data.shape[0]
                else:
                    Start_Col = 0
                    Start_Row = 0
                pred_acc.to_excel(writer, sheet_name="result", encoding='utf-8', startcol=0, startrow=Start_Row,
                                  index=False,header=False)
                writer.save()
            To_Excel_Result(real=real, pred=pred)

        # To_Model_Png(model)
    # ######################################           对整个代码做循环处理              ###################
    if case == 2:
        for step in range(0, 10):
            for rate in range(0, 4):
                for node in range(0, 5):

                    time_step = Time_Step[step]
                    learn_rate = Learn_Rate[rate]
                    hidden_node = Hidden_Node[node]

                    X_train_A = []
                    Y_train_A = []

                    # LSTM的输入必须是  n组训练集中有时间index+属性columns 3维数组 ndarray
                    for i in range(time_step, train_data.shape[0]):
                        X_train_A.append(train_scaler[i - time_step:i, :])
                        Y_train_A.append(train_scaler_A[i])
                    # 此处的append之后带有array（），需要用np。array重新定义成一个数组
                    X_train, Y_train = np.array(X_train_A), np.array(Y_train_A)
                    # 定义成一个数组后，还需要调整一下形状
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
                    # print(X_train)

                    ####################################  测试集处理  #####################################
                    test_data = data.loc[7000 - time_step + 1:]
                    test_data = scaler.transform(test_data)
                    X_test = []
                    Y_test = []
                    for i in range(time_step, test_data.shape[0]):
                        X_test.append(test_data[i - time_step:i, :])
                        Y_test.append(test_data[i - time_step:i, 28])
                    # 取29列就是A组出风口的NOx的输出值
                    X_test = np.array(X_test)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
                    X_test = np.append(X_train, X_test, 0)

                    #####################################模型训练+打开####################################

                    if case == 2:
                        # 训练模型
                        model = LSTM(X_train, Y_train)
                        # learn_rate=0.001
                        # model = LSTM(X_train, Y_train)
                        # learn_rate = 0.01
                        # model = LSTM(X_train, Y_train)
                        # learn_rate = 0.1
                        # model = LSTM(X_train, Y_train)
                        # 保存现有模型
                        # model.save(model_save_path)
                        # 删除当前已存在的模型
                        # del model
                    if case == 2:
                        # 加载模型
                        # model = keras.models.load_model(model_save_path, compile=False, custom_objects={'RMSE': RMSE})
                        #  model = load_model(model_save_path)
                        # 测试集来预测相应的NOx输出
                        pred = model.predict(X_test)
                        # 对于pred进行反归一化，由于scaler的大小为[:,30],因此进行了一步转化；见函数
                        pred = Inverse_Maxmin(pred, scaler)
                        # pred=np.array(pred)
                        pred = pred.reshape(pred.shape[0], 1)
                        # 取出相应的测试集的真实输出
                        real = data.loc[data.shape[0] - pred.shape[0]:, ["Unnamed: 28"]]
                        real = np.array(real)
                        # 可视化
                        # visualization(real=real, pred=pred)
                        pred_acc = pd.DataFrame((abs(real - pred) / real))
                        pred_acc_data = 0
                        for i in range(0, pred_acc.shape[0] - 1):
                            pred_acc_data += pred_acc.iloc[i, 0]
                        pred_acc = pd.DataFrame({'1': [time_step], '2': [learn_rate], '3': [hidden_node],
                                                 'acc': [pred_acc_data / pred_acc.shape[0]]})  # 行求和

                        with pd.ExcelWriter(train_history_path, mode='a', engine='openpyxl',
                                            if_sheet_exists="overlay") as writer:
                            if writer.mode == 'r+b' and writer.if_sheet_exists == 'overlay':
                                excel_data = pd.read_excel(train_history_path, sheet_name="result")
                                Start_Col = excel_data.shape[1]
                                Start_Row = excel_data.shape[0]
                            else:
                                Start_Col = 0
                                Start_Row = 0
                            pred_acc.to_excel(writer, sheet_name="result", encoding='utf-8', startcol=0,
                                              startrow=Start_Row+1,
                                              index=False, header=False)
                            writer.save()
                        #To_Excel_Result(real=real, pred=pred)

                        # To_Model_Png(model)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')