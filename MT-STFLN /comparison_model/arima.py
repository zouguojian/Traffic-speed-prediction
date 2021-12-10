# -- coding: utf-8 --


'''
arima 的predict和forecast的使用可以参考： https://www.cnpython.com/qa/88282

for t in range(len(test)):
    model = ARIMA(history, order=order)
    model_fit = model.fit(disp=-1)
    yhat_f = model_fit.forecast()[0][0]
    yhat_p = model_fit.predict(start=len(history), end=len(history))[0]
    predictions_f.append(yhat_f)
    predictions_p.append(yhat_p)
    history.append(test[t])


for t in range(len(test)):
    model_f = ARIMA(history_f, order=order)
    model_p = ARIMA(history_p, order=order)
    model_fit_f = model_f.fit(disp=-1)
    model_fit_p = model_p.fit(disp=-1)
    yhat_f = model_fit_f.forecast()[0][0]
    yhat_p = model_fit_p.predict(start=len(history_p), end=len(history_p))[0]
    predictions_f.append(yhat_f)
    predictions_p.append(yhat_p)
    history_f.append(yhat_f)
    history_f.append(yhat_p)

model = ARIMA(history, order=order)
    model_fit = model.fit(disp=-1)
    predictions_f_ms = model_fit.forecast(steps=len(test))[0]
    predictions_p_ms = model_fit.predict(start=len(history), end=len(history)+len(test)-1)
'''

import seaborn as sns
sns.set_style("whitegrid",{"font.sans-serif":['KaiTi', 'Arial']})

import pandas as pd
import numpy as np
from gcn_model.hyparameter import parameter
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from gcn_model.data_read import *
sns.set_style("whitegrid", {"font.sans-serif": ['KaiTi', 'Arial']})

para = parameter(argparse.ArgumentParser())
para = para.get_para()

import warnings
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

file='/Users/guojianzou/Traffic-speed-prediction/data/data_hour/'


def accuracy(label, predict):
    '''
    :param Label: represents the observed value
    :param Predict: represents the predicted value
    :param epoch:
    :param steps:
    :return:
    '''
    error = label - predict
    average_error = np.mean(np.fabs(error.astype(float)))
    print("mae is : %.6f" % (average_error))

    rmse_error = np.sqrt(np.mean(np.square(label - predict)))
    print("rmse is : %.6f" % (rmse_error))

    cor = np.mean(np.multiply((label - np.mean(label)),
                              (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
    print('correlation coefficient is: %.6f' % (cor))

    # mask = label != 0
    # mape =np.mean(np.fabs((label[mask] - predict[mask]) / label[mask]))*100.0
    # mape=np.mean(np.fabs((label - predict) / label)) * 100.0
    # print('mape is: %.6f %' % (mape))
    sse = np.sum((label - predict) ** 2)
    sst = np.sum((label - np.mean(label)) ** 2)
    R2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    print('r^2 is: %.6f' % (R2))

    return average_error, rmse_error, cor, R2

labels=[]
predicts=[]

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.9)
    train, test = X[:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    prediction = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        prediction.append(yhat)
        history.append(test[t])
    #
    # model = ARIMA(train, order=arima_order)
    # arima = model.fit(disp=0)
    # prediction = arima.predict(start=len(train), end=len(train)+len(test)-1)[0]
    # print('the prediction shape is : ',prediction.shape)
    #
    # print(prediction, prediction.shape)

    # calculate out of sample error
    error = mean_squared_error(test, prediction)
    return error, test,prediction


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    label, predict = 0, 0
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse, test,prediction = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        label = test
                        predict = prediction
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return label,predict


# load dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


if __name__ == "__main__":
    # series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    data = pd.read_csv(file+'train.csv', encoding='utf-8')  # 数据读取
    # print(data.keys())
    # print(data.values.shape[0] * 0.9)
    #
    #
    # data=data['speed']
    # print(data.shape)
    # print(data.sum())
    # print()
    # median=data.median()
    # min=data.min()
    # max=data.max()


    # evaluate parameters
    # p_values = [1, 2, 4, 6]
    # d_values = range(0, 3)
    # q_values = range(0, 3)

    p_values = [4]
    d_values = [0]
    q_values = [2]
    warnings.filterwarnings("ignore")
    for site in range(1):
        print(site)
        series=data[(data['in_id']==data.values[site][0]) & (data['out_id']==data.values[site][1])]
        print(series.shape)

        label, predict=evaluate_models(series.values[:,-2], p_values, d_values, q_values)
        labels.append(label)
        predicts.append(predict)

    labels=np.reshape(np.array(labels),newshape=[-1])
    predicts=np.reshape(np.array(predicts),newshape=[-1])
    print(labels.shape,predicts.shape)
    accuracy(labels,predicts)