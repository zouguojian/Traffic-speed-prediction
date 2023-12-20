# -- coding: utf-8 --
from models.inits import *
from models.hyparameter import parameter
para = parameter(argparse.ArgumentParser())
para = para.get_para()
file ='/Users/guojianzou/Traffic-speed-prediction/3S-TBLN/data/metr-la/train_5.csv'
def seq2instance(data, P, Q, low_index=0, high_index=100, granularity=15, sites = 108, type='train'):
    '''
    :param data:
    :param P:
    :param Q:
    :param low_index:
    :param high_index:
    :param granularity:
    :param sites:
    :param type:
    :return: (N, sites, P) (N, sites, P+Q) (N, sites, P+Q) (N, sites, P+Q) (N, sites, P+Q) (N, 207, 24) (N, sites, P+Q)
    '''
    X, DoW, D, H, M, L, XAll = [],[],[],[],[],[],[]
    total_week_len = 60//(granularity) * 24 * 7
    while low_index + P + Q <= high_index:
        label = data[low_index * sites: (low_index + P + Q) * sites, -1:]
        label = np.concatenate([label[i * sites: (i + 1) * sites] for i in range(Q+P)], axis=1)
        date = data[low_index * sites: (low_index + P + Q) * sites, 1]
        X.append(np.reshape(data[low_index * sites: (low_index + P) * sites, 5:6],[1, P, sites]))
        DoW.append(np.reshape([datetime.date(int(char.replace('/', '-').split('-')[0]), int(char.replace('/', '-').split('-')[1]),
                              int(char.replace('/', '-').split('-')[2])).weekday() for char in date],[1, P+Q, sites]))
        D.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 2],[1, P+Q, sites]))
        H.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 3],[1, P+Q, sites]))
        hours_to_minutes = data[low_index * sites: (low_index + P + Q) * sites, 3] * 60
        minutes_index_of_day = np.add(hours_to_minutes, data[low_index * sites: (low_index + P + Q) * sites, 4])
        M.append(np.reshape(minutes_index_of_day // granularity,[1, P+Q, sites]))
        L.append(np.reshape(label,[1, sites, Q+P]))
        XAll.append(np.reshape(data[(low_index - total_week_len) * sites: (low_index - total_week_len + P + Q) * sites, 5:6],[1, P+Q, sites]))

        if type =='train':
            low_index += 1
        else:
            low_index += 1

    return np.concatenate(X,axis=0), \
           np.concatenate(DoW,axis=0), \
           np.concatenate(D,axis=0), \
           np.concatenate(H,axis=0), \
           np.concatenate(M,axis=0), \
           np.concatenate(L,axis=0), \
           np.concatenate(XAll,axis=0)


def loadData(args):
    # Traffic
    df = pd.read_csv(file)
    min, max = df['speed'].values.min(), df['speed'].values.max()
    Traffic = df.values
    # train/val/test
    total_samples = df.shape[0]//args.site_num

    train_low = 60 //(args.granularity) * 24 * 7
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.validate_ratio) * total_samples)

    # X, Y, day of week, day, hour, minute, label, all X
    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll = seq2instance(Traffic,
                                                                                   P=args.input_length,
                                                                                   Q=args.output_length,
                                                                                   low_index=train_low,
                                                                                   high_index=val_low,
                                                                                   granularity=args.granularity,
                                                                                   sites=args.site_num,
                                                                                   type='train')
    print('training dataset has been loaded!')
    valX, valDoW, valD, valH, valM, valL, valXAll = seq2instance(Traffic,
                                                                     args.input_length,
                                                                     args.output_length,
                                                                     low_index=val_low,
                                                                     high_index=test_low,
                                                                     granularity=args.granularity,
                                                                     sites=args.site_num,
                                                                     type='validation')
    print('validation dataset has been loaded!')
    testX, testDoW, testD, testH, testM, testL, testXAll = seq2instance(Traffic,
                                                                            args.input_length,
                                                                            args.output_length,
                                                                            low_index=test_low,
                                                                            high_index=total_samples,
                                                                            granularity=args.granularity,
                                                                            sites=args.site_num,
                                                                            type='test')
    print('testing dataset has been loaded!')
    # normalization
    trainX, trainXAll = (trainX  - min) / (max - min), (trainXAll  - min) / (max - min)
    valX, valXAll = (valX  - min) / (max - min), (valXAll  - min) / (max - min)
    testX, testXAll = (testX  - min) / (max - min), (testXAll  - min) / (max - min)

    return (trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll,
            valX, valDoW, valD, valH, valM, valL, valXAll,
            testX, testDoW, testD, testH, testM, testL, testXAll,
            min, max)

# trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, mean, std = loadData(para)
# print(trainX.shape, trainDoW.shape, trainD.shape, trainH.shape, trainM.shape, trainL.shape, trainXAll.shape)
# print(trainX.shape, valX.shape, testX.shape)