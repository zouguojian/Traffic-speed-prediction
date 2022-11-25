# Traffic-speed-prediction

## Model Item
### 1. MT-STGIN [[codes link]](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN%20)
>Multi-task learning Network for Highway Traffic Speed Prediction

### 2. ST-ANet [[codes link]](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/ST-ANet)
>Spatiotemporal Attention Learning Network

### 3. RST-Net [[codes link]](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/ST-ANet)
>Spatiotemporal Encoder-Decoder Neural Network


MT-STGIN has been submitted to IEEE Transactions on Intelligent Transportation Systems TITS journal!!! 
ST-ANet has been accepted in Computer Engineering;
Both codes have been uploaded in the Git-Hub page


1. For MT-STFLN and baselines, the HyperParameters setting as followings:

        self.parser.add_argument('--save_path', type=str, default='gcn/', help='save path')
        # you can select one path address used to save the model's parameters;
        # and modify the default path name, such as default='gcn/'.

        self.parser.add_argument('--data_divide', type=float, default=0.7, help='data_divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epochs', type=int, default=100, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.3, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=49, help='total number of road')

        self.parser.add_argument('--emb_size', type=int, default=256, help='embedding size')
        self.parser.add_argument('--features', type=int, default=1, help='numbers of the feature')
        self.parser.add_argument('--features_p', type=int, default=15, help='numbers of the feature pollution')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=6, help='input length')
        self.parser.add_argument('--output_length', type=int, default=3, help='output length')

        self.parser.add_argument('--model_name', type=str, default='gcn', help='model string')
        # you can change the model name to training or testing the model,
        # when the traning and testing stage before, you should set the defualt model name, that is {default='gcn'}:
        # first step is training, and second step is testing. training stage, you should input number 1, testing stage, you should input number 1.
        # if you have any problems you count, please do not hesitate to contact me, my e-mail address is: 2010768@tongji.edu.cn.
        
        self.parser.add_argument('--hidden1', type=int, default=32, help='number of units in hidden layer 1')
        self.parser.add_argument('--gcn_output_size', type=int, default=64, help='model string')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')

        self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=1.0, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.0, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=1.0, help='test set rate')
