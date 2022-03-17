import pandas as pd
import numpy as np
import datetime

path = 'dragon_flow.csv'

raw_data = pd.read_csv(path,encoding='utf-8')
print('before:',raw_data)
raw_data['date'] = pd.to_datetime(raw_data['date'], format='%Y/%m/%d')

# G000664001004010010

G000664001004010010_DATA = raw_data[raw_data['station'] == "G000664001004010010"]
start_time = datetime.datetime(2021, 7, 13)
end_time = datetime.datetime(2021, 7, 31)
start_time_empty = datetime.datetime(2021, 8, 13)
end_time_empty = datetime.datetime(2021, 8, 31)
# 填入数据
G000664001004010010_DATA_RANGE = G000664001004010010_DATA[(G000664001004010010_DATA['date'] >= start_time)
                                                          & (G000664001004010010_DATA['date'] <= end_time)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001004010010") & (G000664001004010010_DATA['date'] >= start_time_empty)
             & (G000664001004010010_DATA[
                    'date'] <= end_time_empty), 'flow'] = G000664001004010010_DATA_RANGE.values

# raw_data.to_csv('/Users/dongpingping/Documents/博士研究生/出入口流量预测/data/dragon_flow_noempty.csv')

# G008564001000210010  2021/07/16 2021/07/20

G008564001000210010_DATA = raw_data[raw_data['station'] == "G008564001000210010"]

G008564001000210010_time_start = datetime.datetime(2021, 8, 16)
G008564001000210010_time_end = datetime.datetime(2021, 8, 20)
# 缺的
G008564001000210010_start_empty = datetime.datetime(2021, 7, 16)
G008564001000210010_end_empty = datetime.datetime(2021, 7, 20)
G008564001000210010_DATA_RANGE = \
    G008564001000210010_DATA[(G008564001000210010_DATA['date'] >= G008564001000210010_time_start)
                             & (G008564001000210010_DATA['date'] <= G008564001000210010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G008564001000210010") & (
            G008564001000210010_DATA['date'] >= G008564001000210010_start_empty)
             & (G008564001000210010_DATA[
                    'date'] <= G008564001000210010_end_empty), 'flow'] = G008564001000210010_DATA_RANGE.values

# G008564001000220010

G008564001000220010_DATA = raw_data[raw_data['station'] == "G008564001000220010"]

G008564001000220010_time_start = datetime.datetime(2021, 8, 16)
G008564001000220010_time_end = datetime.datetime(2021, 8, 20)
# 缺的
G008564001000220010_start_empty = datetime.datetime(2021, 7, 16)
G008564001000220010_end_empty = datetime.datetime(2021, 7, 20)
G008564001000220010_DATA_RANGE = \
    G008564001000220010_DATA[(G008564001000220010_DATA['date'] >= G008564001000220010_time_start)
                             & (G008564001000220010_DATA['date'] <= G008564001000220010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G008564001000220010") & (
            G008564001000220010_DATA['date'] >= G008564001000220010_start_empty)
             & (G008564001000220010_DATA[
                    'date'] <= G008564001000210010_end_empty), 'flow'] = G008564001000220010_DATA_RANGE.values

# G002064001000320010  2021/07/01 2021/07/23
G002064001000320010_DATA = raw_data[raw_data['station'] == "G002064001000320010"]
G002064001000320010_time_start = datetime.datetime(2021, 8, 1)
G002064001000320010_time_end = datetime.datetime(2021, 8, 23)
# 缺的
G002064001000320010_start_empty = datetime.datetime(2021, 7, 1)
G002064001000320010_end_empty = datetime.datetime(2021, 7, 23)

G002064001000320010_DATA_RANGE = \
    G002064001000320010_DATA[(G002064001000320010_DATA['date'] >= G002064001000320010_time_start)
                             & (G002064001000320010_DATA['date'] <= G002064001000320010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G002064001000320010") & (
            G002064001000320010_DATA['date'] >= G002064001000320010_start_empty)
             & (G002064001000320010_DATA[
                    'date'] <= G002064001000320010_end_empty), 'flow'] = G002064001000320010_DATA_RANGE.values

# G002064001000310010 2021/07/01 2021/07/23
G002064001000310010_DATA = raw_data[raw_data['station'] == "G002064001000310010"]
G002064001000310010_time_start = datetime.datetime(2021, 8, 1)
G002064001000310010_time_end = datetime.datetime(2021, 8, 23)
# 缺的
G002064001000310010_start_empty = datetime.datetime(2021, 7, 1)
G002064001000310010_end_empty = datetime.datetime(2021, 7, 23)

G002064001000310010_DATA_RANGE = \
    G002064001000310010_DATA[(G002064001000310010_DATA['date'] >= G002064001000310010_time_start)
                             & (G002064001000310010_DATA['date'] <= G002064001000310010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G002064001000310010") & (
            G002064001000310010_DATA['date'] >= G002064001000310010_start_empty)
             & (G002064001000310010_DATA[
                    'date'] <= G002064001000310010_end_empty), 'flow'] = G002064001000310010_DATA_RANGE.values



# G002064001000210010 2021/06/01 2021/06/06

G002064001000210010_DATA = raw_data[raw_data['station'] == "G002064001000210010"]
G002064001000210010_time_start = datetime.datetime(2021, 7, 1)
G002064001000210010_time_end = datetime.datetime(2021, 7, 6)
# 缺的
G002064001000210010_start_empty = datetime.datetime(2021, 6, 1)
G002064001000210010_end_empty = datetime.datetime(2021, 6, 6)

G002064001000210010_DATA_RANGE = \
    G002064001000210010_DATA[(G002064001000210010_DATA['date'] >= G002064001000210010_time_start)
                             & (G002064001000210010_DATA['date'] <= G002064001000210010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G002064001000210010") & (
            G002064001000210010_DATA['date'] >= G002064001000210010_start_empty)
             & (G002064001000210010_DATA[
                    'date'] <= G002064001000210010_end_empty), 'flow'] = G002064001000210010_DATA_RANGE.values

# G002064001000220010 2021/06/01 2021/06/05
G002064001000220010_DATA = raw_data[raw_data['station'] == "G002064001000220010"]
G002064001000220010_time_start = datetime.datetime(2021, 7, 1)
G002064001000220010_time_end = datetime.datetime(2021, 7, 5)
# 缺的
G002064001000220010_start_empty = datetime.datetime(2021, 6, 1)
G002064001000220010_end_empty = datetime.datetime(2021, 6, 5)

G002064001000220010_DATA_RANGE = \
    G002064001000220010_DATA[(G002064001000220010_DATA['date'] >= G002064001000220010_time_start)
                             & (G002064001000220010_DATA['date'] <= G002064001000220010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G002064001000220010") & (
            G002064001000220010_DATA['date'] >= G002064001000220010_start_empty)
             & (G002064001000220010_DATA[
                    'date'] <= G002064001000220010_end_empty), 'flow'] = G002064001000220010_DATA_RANGE.values


# G000664001001510010 2021/06/01 2021/06/13

G000664001001510010_DATA = raw_data[raw_data['station'] == "G000664001001510010"]
G000664001001510010_time_start = datetime.datetime(2021, 7, 1)
G000664001001510010_time_end = datetime.datetime(2021, 7, 13)
# 缺的
G000664001001510010_start_empty = datetime.datetime(2021, 6, 1)
G000664001001510010_end_empty = datetime.datetime(2021, 6, 13)

G000664001001510010_DATA_RANGE = \
    G000664001001510010_DATA[(G000664001001510010_DATA['date'] >= G000664001001510010_time_start)
                             & (G000664001001510010_DATA['date'] <= G000664001001510010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001001510010") & (
            G000664001001510010_DATA['date'] >= G000664001001510010_start_empty)
             & (G000664001001510010_DATA[
                    'date'] <= G000664001001510010_end_empty), 'flow'] = G000664001001510010_DATA_RANGE.values



# G000664001001520010 2021/06/01 2021/06/13
G000664001001520010_DATA = raw_data[raw_data['station'] == "G000664001001520010"]
G000664001001520010_time_start = datetime.datetime(2021, 7, 1)
G000664001001520010_time_end = datetime.datetime(2021, 7, 13)
# 缺的
G000664001001520010_start_empty = datetime.datetime(2021, 6, 1)
G000664001001520010_end_empty = datetime.datetime(2021, 6, 13)

G000664001001520010_DATA_RANGE = \
    G000664001001520010_DATA[(G000664001001520010_DATA['date'] >= G000664001001520010_time_start)
                             & (G000664001001520010_DATA['date'] <= G000664001001520010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001001520010") & (
            G000664001001520010_DATA['date'] >= G000664001001520010_start_empty)
             & (G000664001001520010_DATA[
                    'date'] <= G000664001001520010_end_empty), 'flow'] = G000664001001520010_DATA_RANGE.values

# G000664001001410010 2021/06/01 2021/06/13
G000664001001410010_DATA = raw_data[raw_data['station'] == "G000664001001410010"]
G000664001001410010_time_start = datetime.datetime(2021, 7, 1)
G000664001001410010_time_end = datetime.datetime(2021, 7, 13)
# 缺的
G000664001001410010_start_empty = datetime.datetime(2021, 6, 1)
G000664001001410010_end_empty = datetime.datetime(2021, 6, 13)

G000664001001410010_DATA_RANGE = \
    G000664001001410010_DATA[(G000664001001410010_DATA['date'] >= G000664001001410010_time_start)
                             & (G000664001001410010_DATA['date'] <= G000664001001410010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001001410010") & (
            G000664001001410010_DATA['date'] >= G000664001001410010_start_empty)
             & (G000664001001410010_DATA[
                    'date'] <= G000664001001410010_end_empty), 'flow'] = G000664001001410010_DATA_RANGE.values
# G000664001001420010 2021/06/01 2021/06/13
G000664001001420010_DATA = raw_data[raw_data['station'] == "G000664001001420010"]
G000664001001420010_time_start = datetime.datetime(2021, 7, 1)
G000664001001420010_time_end = datetime.datetime(2021, 7, 13)
# 缺的
G000664001001420010_start_empty = datetime.datetime(2021, 6, 1)
G000664001001420010_end_empty = datetime.datetime(2021, 6, 13)

G000664001001420010_DATA_RANGE = \
    G000664001001420010_DATA[(G000664001001420010_DATA['date'] >= G000664001001420010_time_start)
                             & (G000664001001420010_DATA['date'] <= G000664001001420010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001001420010") & (
            G000664001001420010_DATA['date'] >= G000664001001420010_start_empty)
             & (G000664001001420010_DATA[
                    'date'] <= G000664001001420010_end_empty), 'flow'] = G000664001001420010_DATA_RANGE.values


# G000664001000910010 2021/06/01 2021/07/21
G000664001000910010_DATA = raw_data[raw_data['station'] == "G000664001000910010"]
G000664001000910010_time_start = datetime.datetime(2021, 8, 1)
G000664001000910010_time_end = datetime.datetime(2021, 8, 21)
# 缺的
G000664001000910010_start_empty = datetime.datetime(2021, 7, 1)
G000664001000910010_end_empty = datetime.datetime(2021, 7, 21)

G000664001000910010_DATA_RANGE = \
    G000664001000910010_DATA[(G000664001000910010_DATA['date'] >= G000664001000910010_time_start)
                             & (G000664001000910010_DATA['date'] <= G000664001000910010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001000910010") & (
            G000664001000910010_DATA['date'] >= G000664001000910010_start_empty)
             & (G000664001000910010_DATA[
                    'date'] <= G000664001000910010_end_empty), 'flow'] = G000664001000910010_DATA_RANGE.values

G000664001000910010_DATA = raw_data[raw_data['station'] == "G000664001000910010"]

G000664001000910010_time_start2 = datetime.datetime(2021, 7, 1)
G000664001000910010_time_end2 = datetime.datetime(2021, 7, 30)
# 缺的
G000664001000910010_start_empty2 = datetime.datetime(2021, 6, 1)
G000664001000910010_end_empty2 = datetime.datetime(2021, 6, 30)
G000664001000910010_DATA_RANGE2 = \
    G000664001000910010_DATA[(G000664001000910010_DATA['date'] >= G000664001000910010_time_start2)
                             & (G000664001000910010_DATA['date'] <= G000664001000910010_time_end2)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001000910010") & (
            G000664001000910010_DATA['date'] >= G000664001000910010_start_empty2)
             & (G000664001000910010_DATA[
                    'date'] <= G000664001000910010_end_empty2), 'flow'] = G000664001000910010_DATA_RANGE2.values
# G000664001000920010 2021/06/01 2021/07/21
G000664001000920010_DATA = raw_data[raw_data['station'] == "G000664001000920010"]
G000664001000920010_time_start = datetime.datetime(2021, 8, 1)
G000664001000920010_time_end = datetime.datetime(2021, 8, 21)
# 缺的
G000664001000920010_start_empty = datetime.datetime(2021, 7, 1)
G000664001000920010_end_empty = datetime.datetime(2021, 7, 21)

G000664001000920010_DATA_RANGE = \
    G000664001000920010_DATA[(G000664001000920010_DATA['date'] >= G000664001000920010_time_start)
                             & (G000664001000920010_DATA['date'] <= G000664001000920010_time_end)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001000920010") & (
            G000664001000920010_DATA['date'] >= G000664001000920010_start_empty)
             & (G000664001000920010_DATA[
                    'date'] <= G000664001000920010_end_empty), 'flow'] = G000664001000920010_DATA_RANGE.values

G000664001000920010_DATA = raw_data[raw_data['station'] == "G000664001000920010"]

G000664001000920010_time_start2 = datetime.datetime(2021, 7, 1)
G000664001000920010_time_end2 = datetime.datetime(2021, 7, 30)
# 缺的
G000664001000920010_start_empty2 = datetime.datetime(2021, 6, 1)
G000664001000920010_end_empty2 = datetime.datetime(2021, 6, 30)
G000664001000920010_DATA_RANGE2 = \
    G000664001000920010_DATA[(G000664001000920010_DATA['date'] >= G000664001000920010_time_start2)
                             & (G000664001000920010_DATA['date'] <= G000664001000920010_time_end2)]['flow']

raw_data.loc[(raw_data['station'] == "G000664001000920010") & (
            G000664001000920010_DATA['date'] >= G000664001000920010_start_empty2)
             & (G000664001000920010_DATA[
                    'date'] <= G000664001000920010_end_empty2), 'flow'] = G000664001000920010_DATA_RANGE2.values
# print(raw_data.loc[(raw_data['station'] == "G000664001000910010") & (
#             G000664001000920010_DATA['date'] >= G000664001000920010_start_empty2)
#              & (G000664001000920010_DATA[
#                     'date'] <= G000664001000920010_end_empty2), 'flow'])
print('after:',raw_data)
raw_data.to_csv('dragon_flow_fill.csv')