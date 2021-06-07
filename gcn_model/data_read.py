# -- coding: utf-8 --
import pandas as pd
import csv

file_path=r'/Users/guojianzou/Documents/同济大学/智慧公路云平台/OD/data/fivemonth_qiepian.csv'
save_path=r'/Users/guojianzou/Documents/同济大学/智慧公路云平台/OD/data/data_all.csv'

train_path=r'/Users/guojianzou/Documents/同济大学/智慧公路云平台/OD/data/train.csv'


def create_train(year,month,file_path, save_path, train_path):
    data = pd.read_csv(file_path, encoding='gb2312')
    new_key = ['IN_ID', 'OUT_ID', 'Year', 'Month', 'Day', 'Hour', 'Speed']
    id_dict=dict()

    file = open(save_path, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(new_key)
    for line in data.values:
        if (line[0],line[1]) not in id_dict and line[0]!=line[1]: id_dict[(line[0],line[1])]=1
    id_dict=sorted(id_dict)
    print(id_dict)

    for line in data.values:  #save
        if line[0]==line[1]:continue
        new_line=list()
        for i in range(len(line)):
            if i==2:
                for ch in line[2].split('/'):
                    new_line.append(int(ch))
            else:
                if i==len(line)-1:new_line.append(float(line[i]))
                else:new_line.append(int(line[i]))
        writer.writerow(new_line)
    file.close()

    data=pd.read_csv(save_path)  #create_train
    file = open(train_path, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(new_key)

    for d in range(1,32):
        data1=data.loc[data['Day'] == d]
        if data1.values.shape[0]==0:
            print('day empty')
            continue
        for h in range(0,24):
            data2 = data1.loc[data1['Hour'] == h]
            if data2.values.shape[0] == 0:
                print('hour empty')
                for id in id_dict:
                    line = [id[0], id[1], year, month, d, h, 0.0]
                    writer.writerow(line)
                continue
            for id in id_dict:
                data3 = data2.loc[(data2['IN_ID'] == id[0]) & (data2['OUT_ID'] == id[1])]
                if data3.values.shape[0] == 0:
                    print('zone empty')
                    line = [id[0], id[1], year, month, d, h, 0]
                    writer.writerow(line)
                    continue
                print(data3.values.shape)
                line=[id[0], id[1], year, month, d, h, data3.values[-1,-1]]
                writer.writerow(line)
    file.close()

if __name__=='__main__':
    print('!!!........................beginning.........................!!!')
    create_train(2021,5,file_path,save_path,train_path)
    print('!!!........................finishing.........................!!!')