# -- coding: utf-8 --
import pandas as pd
import csv
import os

data_path=r'data.csv'
save_path=r'filter_data.csv'
train_path=r'train.csv'
delete_path=r'delete_path.csv'

data=pd.read_csv(r'data.csv', encoding='gb2312')
print(data.values.shape)

new_key=['in_id', 'out_id', 'year', 'month', 'day', 'hour', 'speed', 'flow']
month=[31,28,31,30,31,30,31,31,30,31,30,31]

def delete(file_path=None, delete_path=None):
    '''
    :param file_path:
    :return:
    '''

    key = ['in_id', 'out_id', 'year', 'hour', 'speed', 'flow']

    if not os.path.exists(file_path):
        pass
    else:
        data=pd.read_csv(file_path,encoding='gb2312')

        file = open(delete_path, 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(key)

        for line in data.values:
            if line[0]==line[1]:continue
            else:
                # y_m_d=line[2].split('/')
                # if len(y_m_d[1])==1:y_m_d[1]='0'+y_m_d[1]
                # if len(y_m_d[2])==1:y_m_d[2]='0'+y_m_d[2]
                #
                # y_m_d[1]='/'+y_m_d[1]
                # y_m_d[2] = '/' + y_m_d[2]
                #
                # line[2]=''.join(y_m_d)

                writer.writerow(line)
        file.close()

# delete(file_path=data_path,delete_path=delete_path)


def day_empty(writer=None, id_dict=None, year=2020, month=5, day=1):
    '''
    :param writer:
    :param year:
    :param month:
    :param day:
    :return:
    '''
    for h in range(0, 24):
        for id in id_dict:
            line = [id[0], id[1], year, month, day, h, 0.0, 0]
            writer.writerow(line)

def hour_empty(writer=None, id_dict=None,year=2020, month=5, day=1, hour=0):
    '''
    :param writer:
    :param id_dict:
    :param year:
    :param month:
    :param day:
    :param hour:
    :return:
    '''
    for id in id_dict:
        line = [id[0], id[1], year, month, day, hour, 0.0, 0]
        writer.writerow(line)

def min_empty(writer=None, id_dict=None,year=2020, month=5, day=1, hour=0, min=15):
    '''
    :param writer:
    :param id_dict:
    :param year:
    :param month:
    :param day:
    :param hour:
    :param min:
    :return:
    '''
    for id in id_dict:
        line = [id[0], id[1], year, month, day, hour, min, 0.0, 0]
        writer.writerow(line)

def save_train(file_path=None, train_path=None, year=2020):
    if not os.path.exists(file_path):
        pass
    else:
        data = pd.read_csv(file_path)  # create_train
        id_dict=dict()
        for line in data.values:
            if (line[0], line[1]) not in id_dict and line[0] != line[1]: id_dict[(line[0], line[1])] = 1
        id_dict = sorted(id_dict)

        file = open(train_path, 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(new_key)

        for m in range(5,9):
            # day
            for d in range(1, month[m-1]+1):
                data1 = data.loc[data['year'] == (str(year)+'/'+str(m)+'/'+str(d))]
                if data1.values.shape[0] == 0:
                    print('day empty')
                    day_empty(writer=writer,id_dict=id_dict, year=year,month=m,day=d)
                    continue
            # hour
                for h in range(0, 24):
                    data2 = data1.loc[data1['hour'] == h]
                    if data2.values.shape[0] == 0:
                        print('hour empty')
                        hour_empty(writer=writer,id_dict=id_dict,year=year,month=m, day=d, hour=h)
                        continue
            # road
                    for id in id_dict:
                        data3 = data2.loc[(data2['in_id'] == id[0]) & (data2['out_id'] == id[1])]
                        if data3.values.shape[0] == 0:
                            print('road empty')
                            line = [id[0], id[1], year, m, d, h, 0.0, 0]
                            writer.writerow(line)
                            continue

                        print(data3.values)
                        line = [id[0], id[1], year, m, d, h, data3.values[-1, -2],data3.values[-1, -1]]
                        writer.writerow(line)
        file.close()

if __name__=='__main__':
    print('!!!........................beginning.........................!!!')
    save_train(file_path=delete_path,train_path=train_path,year=2020)
    print('!!!........................finishing.........................!!!')