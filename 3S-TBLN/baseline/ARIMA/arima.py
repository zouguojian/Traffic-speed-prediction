# -- coding: utf-8 --
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime

import warnings

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        # mae = np.nan_to_num(mae * mask)
        # wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        # rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    return mae, rmse, mape

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order,k):
    # prepare training dataset
    train_size = int(len(X) * 0.85)
    train, test = X[train_size-20: train_size], X[train_size: ]
    history = [x for x in train]
    # test = [x for x in test]
    # make predictions
    prediction = list()
    test_list=list()
    for t in range(0, len(test)-k, k):
        try:
            # print('t is : ',t)
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast(steps=k)[0]
            prediction.append(yhat)
            test_list.append(test[t:t+k])
            # history.append(test[t:t+k+1])
            history = history[k:]+[char for char in test[t:t+k]]
        except:
            pass
    return test_list, prediction


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values,k):
    dataset = dataset.astype('float32')
    test, prediction = evaluate_arima_model(dataset, [p_values, d_values, q_values],k)
    return test,prediction

if __name__ == "__main__":
    # series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    data = pd.read_csv('train_5.csv', encoding='utf-8')  # 数据读取
    # evaluate parameters
    # p_values = [1, 2, 4, 6]
    # d_values = range(0, 3)
    # q_values = range(0, 3)

    warnings.filterwarnings("ignore")
    start_time = datetime.datetime.now()
    labels = []
    predicts = []
    for site in range(0, 207): # 30 no problem
        print('site is , ',site)
        series=data.loc[data['node'] == site]
        label, predict=evaluate_models(series.values[:,-1], p_values=1, d_values=0, q_values=0, k=12)
        if len(label)>1:
            labels.append(label)
            predicts.append(predict)

    labels=np.concatenate(labels,axis=0)
    predicts=np.concatenate(predicts, axis=0)
    print(labels.shape, predicts.shape)
    print('                MAE\t\tRMSE\t\tMAPE')
    for time_step in range(12):
        mae, rmse, mape = metric(pred= predicts[:,time_step],label=labels[:,time_step])
        print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (time_step + 1, mae, rmse, mape * 100))
    mae, rmse, mape = metric(pred=predicts,label=labels)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total running times is : %f" % total_time.total_seconds())