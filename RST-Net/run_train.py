# -- coding: utf-8 --
'''
the shape of sparsetensor is a tuuple, like this
(array([[  0, 297],
       [  0, 296],
       [  0, 295],
       ...,
       [161,   2],
       [161,   1],
       [161,   0]], dtype=int32), array([0.00323625, 0.00485437, 0.00323625, ..., 0.00646204, 0.00161551,
       0.00161551], dtype=float32), (162, 300))
axis=0: is nonzero values, x-axis represents Row, y-axis represents Column.
axis=1: corresponding the nonzero value.
axis=2: represents the sparse matrix shape.
'''

from __future__ import division
from __future__ import print_function
from models.utils import *
from models.inits import *
from models.models import GCN
from models.hyparameter import parameter
from models.embedding import embedding
from models.encoder import EncoderST
from models.semantic_transformer import SemanticTransformer
from models.bridge import BridgeTransformer
from models.inference import InferenceClass
from models.data_next import DataClass

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class Model(object):
    def __init__(self, para):
        self.para = para
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.emb_size = self.para.emb_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.divide_ratio = self.para.divide_ratio
        self.model_name = self.para.model_name
        self.adj = preprocess_adj(self.adjecent())

        # define gcn model
        if self.para.model_name == 'gcn_cheby':
            self.support = chebyshev_polynomials(self.adj, self.para.max_degree)
            self.num_supports = 1 + self.para.max_degree
            self.model_func = GCN
        else:
            self.support = [self.adj]
            self.num_supports = 1
            self.model_func = GCN

        # placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.para.site_num), name='input_position'),
            'week': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='day_of_week'),
            'day': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_minute'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num, self.features],name='inputs'),
            'features_all': tf.placeholder(tf.float32, shape=[None, self.input_len+self.output_len, self.site_num, self.features],name='total_inputs'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.total_len], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'random_mask': tf.placeholder(tf.int32, shape=(None, self.site_num, self.total_len), name='mask'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero')  # helper variable for sparse dropout
        }
        self.supports = [tf.SparseTensor(indices=self.placeholders['indices_i'],
                                         values=self.placeholders['values_i'],
                                         dense_shape=self.placeholders['dense_shape_i']) for _ in range(self.num_supports)]
        self.embeddings()
        self.model()

    def adjecent(self):
        '''
        :return: adj matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.para.file_adj)
        adj = np.zeros(shape=[self.site_num, self.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def embeddings(self):
        '''
        :return:
        '''
        p_emd = embedding(self.placeholders['position'], vocab_size=self.site_num, num_units=self.para.emb_size, scale=False, scope="position_embed")
        p_emd = tf.reshape(p_emd, shape=[1, self.site_num, self.emb_size])
        self.p_emd = tf.tile(tf.expand_dims(p_emd, axis=0), [self.batch_size, self.total_len, 1, 1])

        w_emb = embedding(self.placeholders['week'], vocab_size=7, num_units=self.emb_size, scale=False,
                          scope="day_of_week_embed")
        self.w_emd = tf.reshape(w_emb, shape=[self.batch_size, self.total_len, self.site_num, self.emb_size])

        d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.emb_size, scale=False, scope="day_embed")
        self.d_emd = tf.reshape(d_emb, shape=[self.batch_size, self.total_len, self.site_num, self.emb_size])

        h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.emb_size, scale=False, scope="hour_embed")
        self.h_emd = tf.reshape(h_emb, shape=[self.batch_size, self.total_len, self.site_num, self.emb_size])

        m_emb = embedding(self.placeholders['minute'], vocab_size=4, num_units=self.para.emb_size, scale=False, scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[self.batch_size, self.total_len, self.site_num, self.emb_size])

    def model(self):
        '''
        :return:
        '''
        '''#................................in the encoder step....................................#'''
        with tf.variable_scope(name_or_scope='encoder', reuse=tf.AUTO_REUSE):
            '''
            return, the gcn output --- for example, inputs.shape is :  (32, 3, 162, 32)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            speed = FC(self.placeholders['features'], units=[self.emb_size, self.emb_size], activations=[tf.nn.relu, None],
                            bn=False, bn_decay=0.99, is_training=self.is_training)
            total_speed = FC(self.placeholders['features_all'], units=[self.emb_size, self.emb_size], activations=[tf.nn.relu, None],
                            bn=False, bn_decay=0.99, is_training=self.is_training)

            speed = tf.add(speed, total_speed[:,:self.input_len])

            # speed = tf.transpose(self.placeholders['features'], perm=[0, 2, 1, 3]) # [-1, input_length, site num, emb_size]
            # semantic_trans = SemanticTransformer(input_len=self.input_len,emb_size=self.emb_size,site_num=self.site_num,features=self.features)
            # speed = semantic_trans.transfer(speed)
            ste = STEmbedding(SE=self.p_emd, TE=[self.w_emd, self.h_emd, self.m_emd], T=0, D=self.emb_size, bn=False, bn_decay=0.99, is_training=self.is_training)

            encoder = EncoderST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func)
            encoder_outs = encoder.encoder_spatio_temporal(speed=speed,
                                                            STE=ste[:, :self.input_len],
                                                            supports=self.supports)
            print('encoder encoder_outs shape is : ', encoder_outs.shape)

        '''#................................in the bridge step.....................................#'''
        with tf.variable_scope(name_or_scope='bridge'):
            # STE[:, :self.para.input_length,:,:]
            # bridge_outs = transformAttention(encoder_outs, encoder_outs, STE[:, self.para.input_length:,:,:], self.para.num_heads, self.para.emb_size // self.para.num_heads, False, 0.99, self.para.is_training)
            bridge = BridgeTransformer(self.para)
            bridge_outs = bridge.encoder(X=encoder_outs,
                                         X_P=encoder_outs,
                                         X_Q=ste[:, self.input_len:]+total_speed[:,self.input_len:])
            print('bridge bridge_outs shape is : ', bridge_outs.shape)

        '''#............................in the reconstruct bridge step.............................#'''
        with tf.variable_scope(name_or_scope='bridge'):
            # STE[:, :self.para.input_length,:,:]
            # bridge_outs = transformAttention(encoder_outs, encoder_outs, STE[:, self.para.input_length:,:,:], self.para.num_heads, self.para.emb_size // self.para.num_heads, False, 0.99, self.para.is_training)
            reconstruct_bridge = BridgeTransformer(self.para)
            reconstruct_bridge_outs = reconstruct_bridge.encoder(X=tf.reverse(bridge_outs, axis=[1]),
                                                                 X_P=tf.reverse(bridge_outs, axis=[1]),
                                                                 X_Q=tf.reverse(ste[:, :self.input_len]+total_speed[:,:self.input_len], axis=[1]))
            print('reconstruction bridge outs shape is : ', reconstruct_bridge_outs.shape)

        '''#...........................in the reconstruction encoder step..........................#'''
        with tf.variable_scope(name_or_scope='encoder', reuse=tf.AUTO_REUSE):
            reconstruct = EncoderST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func)
            reconstruct_outs = reconstruct.encoder_spatio_temporal(speed=reconstruct_bridge_outs,
                                                                   STE=tf.reverse(ste[:, :self.input_len], axis=[1]),
                                                                   supports=self.supports)
            reconstruct_outs = tf.reverse(reconstruct_outs, axis=[1])
            print('reconstruct outs shape is : ', reconstruct_outs.shape)
        reconstruct_outs = tf.concat([reconstruct_outs, bridge_outs], axis=1)

        '''#................................in the inference step.................................#'''
        with tf.variable_scope(name_or_scope='inference'):
            inference = InferenceClass(para=self.para)
            self.pres_s = inference.inference(out_hiddens=reconstruct_outs)
            print('pres_s shape is : ', self.pres_s.shape)


        # observed = tf.where(tf.equal(self.placeholders['random_mask'],1), self.placeholders['labels'], tf.zeros_like(self.placeholders['labels']))
        # predicted = tf.where(tf.equal(self.placeholders['random_mask'],1), self.pres_s, tf.zeros_like(self.placeholders['labels']))
        observed = self.placeholders['labels']
        predicted = self.pres_s
        '''
        它的作用是：对于张量a，如果random_mask对应位置的元素为True，则张量a中的该位置处元素保留，反之由张量b中相应位置的元素来代替。
        '''
        maes_1 = tf.losses.absolute_difference(predicted[:, :, self.input_len:], observed[:, :, self.input_len:])
        self.loss1 = tf.reduce_mean(maes_1)

        maes_2 = tf.losses.absolute_difference(predicted[:, :, :self.input_len], observed[:, :, :self.input_len])
        self.loss2 = tf.reduce_mean(maes_2)

        self.loss = 0.7 * self.loss1 + 0.3 * self.loss2
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        '''#...............................in the training step....................................#'''

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return a * (max - min) + min

    def run_epoch(self):
        max_mae = 100
        self.sess.run(tf.global_variables_initializer())
        iterate = DataClass(self.para)

        train_next = iterate.next_batch(batch_size=self.batch_size, epoch=self.epochs, is_training=True)

        for i in range(int((iterate.length // self.site_num * self.divide_ratio - self.total_len))
                       * self.epochs // self.batch_size):
            random_mask = np.random.randint(low=0,high=2,size=[self.batch_size, self.site_num, self.total_len],dtype=np.int)
            xs, d_of_week, day, hour, minute, labels, xs_all = self.sess.run(train_next)
            xs = np.reshape(xs, [-1, self.input_len, self.site_num, self.features])
            d_of_week = np.reshape(d_of_week, [-1, self.site_num])
            day = np.reshape(day, [-1, self.site_num])
            hour = np.reshape(hour, [-1, self.site_num])
            minute = np.reshape(minute, [-1, self.site_num])
            labels = np.concatenate((np.transpose(np.squeeze(xs, axis=-1), [0, 2, 1]), labels), axis=2)
            xs_all = np.reshape(xs_all, [-1, self.input_len+self.output_len, self.site_num, self.features])
            feed_dict = construct_feed_dict(xs, xs_all, self.adj, labels, d_of_week, day, hour, minute, mask=random_mask, placeholders=self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})
            loss,_ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training total average loss is : %.6f" % (i, loss))

            # validate processing
            if i % 100 == 0:
                mae = self.evaluate()
                if max_mae > mae:
                    print("the validate average loss value is : %.6f" % (mae))
                    max_mae = mae
                    self.saver.save(self.sess, save_path=self.para.save_path + 'model.ckpt')

    def evaluate(self):
        '''
        :return:
        '''
        labels_list, pres_list = list(), list()

        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

        iterate_test = DataClass(hp=self.para)
        test_next = iterate_test.next_batch(batch_size=self.batch_size, epoch=1, is_training=False)
        max_s, min_s = iterate_test.max_s['speed'], iterate_test.min_s['speed']

        file = open('results/'+str(self.model_name)+'.csv', 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(
            ['road'] + ['day_' + str(i) for i in range(self.output_len)] + ['hour_' + str(i) for i in range(
                self.para.output_length)] +
            ['minute_' + str(i) for i in range(self.output_len)] + ['label_' + str(i) for i in
                                                                             range(self.output_len)] +
            ['predict_' + str(i) for i in range(self.output_len)])

        for i in range(int((iterate_test.length // self.site_num - iterate_test.length // self.site_num * self.divide_ratio
                            - self.total_len) // self.output_len) // self.batch_size):
            random_mask = np.ones(shape=[self.batch_size,self.site_num,self.total_len],dtype=np.int)
            xs, d_of_week, day, hour, minute, labels, xs_all= self.sess.run(test_next)
            xs = np.reshape(xs, [-1, self.input_len, self.site_num, self.features])
            d_of_week = np.reshape(d_of_week, [-1, self.site_num])
            day = np.reshape(day, [-1, self.site_num])
            hour = np.reshape(hour, [-1, self.site_num])
            minute = np.reshape(minute, [-1, self.site_num])
            labels = np.concatenate((np.transpose(np.squeeze(xs, axis=-1), [0, 2, 1]), labels), axis=2)
            xs_all = np.reshape(xs_all, [-1, self.input_len + self.output_len, self.site_num, self.features])
            feed_dict = construct_feed_dict(xs, xs_all, self.adj, labels, d_of_week, day, hour, minute, mask=random_mask, placeholders=self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre_s = self.sess.run((self.pres_s), feed_dict=feed_dict)

            for site in range(self.site_num):
                writer.writerow([site]+list(day[self.input_len:,0])+
                                 list(hour[self.input_len:,0])+
                                 list(minute[self.input_len:,0]*15)+
                                 list(np.round(self.re_current(labels[0][site,self.input_len:],max_s,min_s)))+
                                 list(np.round(self.re_current(pre_s[0][site,self.input_len:],max_s,min_s))))

            labels_list.append(labels[:, :, self.input_len:])
            pres_list.append(pre_s[:, :, self.input_len:])

        labels_list = np.reshape(np.array(labels_list, dtype=np.float32),
                                  [-1, self.site_num, self.output_len]).transpose([1, 0, 2])
        pres_list = np.reshape(np.array(pres_list, dtype=np.float32),
                                [-1, self.site_num, self.output_len]).transpose([1, 0, 2])
        if self.para.normalize:
            labels_list = self.re_current(labels_list, max_s, min_s)
            pres_list = self.re_current(pres_list, max_s, min_s)

        print('############# speed prediction result #############')
        mae, rmse, mape, cor, r2 = metric(pres_list, labels_list)  # 产生预测指标
        print('############# speed prediction result #############')
        # for i in range(self.para.output_length):
        #     print('in the %d time step, the evaluating indicator'%(i+1))
        #     metric(pre_s_list[:28,:,i], label_s_list[:28,:,i])

        # describe(label_list, predict_list)   #预测值可视化
        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:
        pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()