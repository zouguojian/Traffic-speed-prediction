# -- coding: utf-8 --
from models.inits import *

class DataClass(object):
    def __init__(self, hp=None):
        '''
        :param hp:
        '''
        self.hp = hp                              # hyperparameter
        self.input_length=self.hp.input_length         # time series length of input
        self.output_length=self.hp.output_length       # the length of prediction
        self.is_training=self.hp.is_training           # true or false
        self.train_ratio=self.hp.train_ratio         # the train ratio between in training set and test set ratio
        self.step=self.hp.step                         # windows step
        self.site_num=self.hp.site_num
        self.granularity = self.hp.granularity
        self.file_train_s= self.hp.file_train_s
        self.normalize = self.hp.normalize             # data normalization

        self.data_s=self.get_source_data(self.file_train_s)
        self.shape_s=self.data_s.shape

        self.length=self.data_s.shape[0]//self.site_num                        # data length
        self.std_s, self.mean_s= self.get_mean_std(self.data_s, ['speed'])   # std and mean values' dictionary
        self.normalization(self.data_s, ['speed'], std_dict=self.std_s, mean_dict=self.mean_s, is_normalize=self.normalize)                  # normalization

    def get_source_data(self,file_path=None):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def get_mean_std(self,data=None, keys=None):
        '''
        :param data:
        :return:
        '''
        mean_dict=dict()
        std_dict=dict()

        for key in keys:
            mean_dict[key] = data[key].mean()
            std_dict[key] = data[key].std()
        # print('the max feature list is :', max_dict)
        # print('the min feature list is :', min_dict)
        return std_dict, mean_dict

    def normalization(self, data, keys=None, std_dict =None, mean_dict=None, is_normalize=True):
        '''
        :param data:
        :param keys:  is a list
        :param is_normalize:
        :return:
        '''
        if is_normalize:
            for key in keys:
                data[key]=(data[key] - mean_dict[key]) / (std_dict[key])

    def generator(self):
        '''
        :return: yield the data of every time, input shape: [batch, site_num*(input_length+output_length)*features]
        label:   [batch, site_num, output_length]
        '''
        data_s = self.data_s.values
        total_week_len=60//(self.granularity) * 24 * 7 # 使用分钟的颗粒度对一个小时60分钟进行分割, granularity可以是5或者15
        # 目的是为了获取过去一个星期在相同时间节点的交通速度情况
        low, high = total_week_len, int(self.length * self.train_ratio)

        while low + self.input_length + self.output_length <= high:
            label=data_s[(low + self.input_length) * self.site_num: (low + self.input_length + self.output_length) * self.site_num,-1:]
            label=np.concatenate([label[i * self.site_num : (i + 1) * self.site_num, :] for i in range(self.output_length)], axis=1)
            date= data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 1]
            yield (data_s[low * self.site_num : (low + self.input_length) * self.site_num, 5:6],
                   [datetime.date(int(char.replace('/','-').split('-')[0]), int(char.replace('/','-').split('-')[1]), int(char.replace('/','-').split('-')[2])).weekday() for char in date],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 2],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 3],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 4]//self.granularity,
                   label,
                   data_s[(low-total_week_len) * self.site_num: (low-total_week_len + self.input_length+self.output_length) * self.site_num, 5:6]
                   )
            low += 1

    def next_batch(self, batch_size, epoch, is_training=True):
        '''
        :param batch_size:
        :param epochs:
        :param is_training:
        :return: x shape is [batch, input_length*site_num, features];
                 day shape is [batch, (input_length+output_length)*site_num];
                 hour shape is [batch, (input_length+output_length)*site_num];
                 minute shape is [batch, (input_length+output_length)*site_num];
                 label shape is [batch, output_length*site_num, features]
        '''
        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32))
        dataset=dataset.shuffle(buffer_size=int(self.length * self.train_ratio-self.input_length-self.output_length)//self.step)
        dataset=dataset.repeat(count=epoch)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()
        return iterator.get_next()


    def test_generator(self):
        '''
        :return: yield the data of every time, input shape: [batch, site_num*(input_length+output_length)*features]
        label:   [batch, site_num, output_length]
        '''
        data_s = self.data_s.values
        total_week_len=60//(self.granularity) * 24 * 7
        low, high = int(self.length * self.train_ratio), int(self.length)

        while low + self.input_length + self.output_length <= high:
            label=data_s[(low + self.input_length) * self.site_num: (low + self.input_length + self.output_length) * self.site_num,-1:]
            label=np.concatenate([label[i * self.site_num : (i + 1) * self.site_num, :] for i in range(self.output_length)], axis=1)
            date= data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 1]
            yield (data_s[low * self.site_num : (low + self.input_length) * self.site_num, 5:6],
                   [datetime.date(int(char.replace('/','-').split('-')[0]), int(char.replace('/','-').split('-')[1]), int(char.replace('/','-').split('-')[2])).weekday() for char in date],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 2],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 3],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 4]//self.granularity,
                   label,
                   data_s[(low-total_week_len) * self.site_num: (low-total_week_len + self.input_length+self.output_length) * self.site_num, 5:6]
                   )
            low += self.hp.predict_steps
            # low += self.output_length

    def test_batch(self, batch_size, epoch, is_training=True):
        '''
        :param batch_size:
        :param epochs:
        :param is_training:
        :return: x shape is [batch, input_length*site_num, features];
                 day shape is [batch, (input_length+output_length)*site_num];
                 hour shape is [batch, (input_length+output_length)*site_num];
                 minute shape is [batch, (input_length+output_length)*site_num];
                 label shape is [batch, output_length*site_num, features]
        '''
        dataset=tf.data.Dataset.from_generator(self.test_generator,output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32))
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()

# #
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataClass(hp=para)
    print(iter.data_s.keys())

    next=iter.next_batch(batch_size=12, epoch=1, is_training=False)
    with tf.Session() as sess:
        for _ in range(4):
            x, d, h, m, y=sess.run(next)
            print(x.shape)
            print(y.shape)
            print(d[0,0],h[0,0],m[0,0])