# TRAFFIC-SPEED-PREDICTION

## WHAT SHOULD WE PAY ATTENTION TO FOCUS ON THE RUNNING ENVIRONMENT?

<font face="微软雅黑" >Note that we need to install the right packages to guarantee the model runs according to the file requirements.txt！！！</font>
  
>* first, please use the conda tool to create a virtual environment, such as ‘conda create traffic speed’;  
> * second, active the environment, and conda activate traffic speed;   
> * third, build environment, the required environments have been added in the file named requirements.txt; you can use conda as an auxiliary tool to install or pip, e.g., conda install tensorflow==1.13.1;    
> * if you have installed the last TensorFlow version, it’s okay; import tensorflow.compat.v1 as tf and tf.disable_v2_behavior();    
> * finally, please click the run_train.py file; the codes are then running;  
> * Note that our TensorFlow version is 1.14.1 and can also be operated on the 2.0. version.  
---
## DATA DESCRIPTION  
> The traffic speed data used in this study is provided by the ETC intelligent monitoring sensors at the gantries and the toll stations of the highway in Yinchuan City, Ningxia Province, China. The 66 ETC intelligent monitoring sensors record the vehicle driving data in real-time, including 13 highway toll stations (each toll station contains an entrance and exit) and 40 highway gantries. Therefore, these monitoring sensors divide the highway network into 108 road segments. The traffic speed of each road segment is measured at a certain frequency, such that one sample is measured every 15 minutes, and therefore the time series form of the traffic speed is obtained. In addition, the traffic speed data also includes the other two factors, timestamps and road segment index. Because of traffic speed heterogeneity on different types of road segments, the traffic speed is divided into three types, from entrance toll to gantry, called ETTG; gantry to gantry, called GTG; and gantry to exit toll, called GTET. The time span is from June 1, 2021, to August 31, 2021. The road segment index does not change over time, and there are 108 road segments in total, that is, 108 indexes. In the experiment, 70% of the data are used as the training set, 15% of the data are used as the validation set, and the remaining 15% are considered as the test set ([dataset link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/data/speed)).
---
## EXPERIMENTAL SETTING  

> The hyperparameters in our MT-STGIN model and base- lines are determined during the training process; that is, the best-performing model is selected according to the MAE on the validation set. Therefore, the validation set used in this study is closely related to the training stage, and after each epoch, the MAE obtained by the prediction model on the validation set is calculated. The specific process is as follows: for each experiment, the number of epochs is 200. After train- ing for an epoch, we test the trained model on the validation set. If the MAE of the prediction model on the validation set decreases, we update and save the model parameters. After many parameter adjustments and experiments, when the prediction effect of the prediction model on the validation set is optimal, the training process ends. Finally, the prediction result is obtained by iterating all the samples in the test set. In all experiments, we use an early-stop mechanism; that is, the number of early-stop rounds and the maximum number of epochs are set to 300 and 50, respectively. To consist with existing studies, we set the target time steps Q and historical time steps P to 6 and 12, respectively, representing the time span is 270 minutes.  

> After multiple training steps, the final model framework parameters are determined. Table I presents the number of layers, nodes, output size and related hyperparameters of the MT-STGIN model. We implement the MT-STGIN and baselines in TensorFlow and PyTorch. The server’s 4 NVIDIA Tesla V100S-PCIE-32GB GPUs and 24 CPU cores are used for model training and testing. Note that the implementation codes of the proposed MT-STGIN model and baseline models are open source, and are available at the personal GitHub homepage [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN).
---
## METRICS

> In order to evaluate the prediction performance of the MT-STGIN model, three metrics are used to determine the difference between the observed values Y and the predicted values Yˆ : the root mean square error (RMSE), mean absolute error (MAE), and mean absolute percentage error (MAPE). Note that low MAE, RMSE, and MAPE values indicate a more accurate prediction performance. 
---

## MT-STGIN AND BASELINES （ALL METHODS' CODES HAVE BEEN REPRODUCED） 
#### HA  [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/ha)
* HA, the HA model uses the average value of the historical data, at the same time every day, as the predicted value at the same time in the future prediction task.  
#### ARIMA [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/arima)
* ARIMA, this is a traditional time series prediction method that combines the moving average and autoregressive components in order to model the historical time series data.
#### SVM [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline)
* SVM, it refers to support vector machine, is a regression technique for short-term prediction of traffic speed.
#### LSTM NN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/lstm)
* LSTM NN, it is used to capture the nonlinear traffic dynamic characteristics.
#### Bi-LSTM NN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/bi_lstm)
* Bi-LSTM NN, it refers to bidirectional long short-term memory neural network. It models each critical path, and then uses the multiple Bi-LSTM layers stacked together in order to merge the time information.
#### FI-RNNs [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/firnn)
* FI-RNNs, it refers to features injected recurrent neural networks. It combines the time series data and uses a stacked RNN and encoder, in order to learn the sequential features of the traffic data.
#### PSPNN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/pspnn)
* PSPNN, it refers to path-based speed prediction neural network. It is composed of a CNN and a bidirectional LSTM (Bi-LSTM) network, that extract the temporal and spatial correlations of the historical data, in order to perform the path- based speed prediction.
#### MDL [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/mdl)
* MDL, it refers to novel mixed deep learning. This method is used to predict the lane-level short-term traffic speed. It consists of a convolutional long and short-term memory (Conv-LSTM) layer, a convolutional layer and a fully connected layer.
#### GMAN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/gman)
* GMAN, it refers to graph multi-attention network. This network is based on spatial and temporal attention. It predicts the traffic speed at different locations on the road network.
#### T-GCN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/tgcn)
* T-GCN, it combines the GCN and GRU to model the spatio-temporal correlations.
#### AST-GAT [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/astgat)
* AST-GAT, it refers to attention-based spatiotemporal graph attention network. It consists of a self-attention-based GAT network and an attention-based LSTM network, for segment-level traffic speed prediction.
#### DCRNN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/dcrnn)
* DCRNN, it is a diffusion convolutional recurrent neural network, a deep learning framework incorporating spatial and temporal dependency into traffic prediction.
#### ST-GRAT [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/st_grat)
* ST-GRAT, it is a novel spatiotemporal graph attention model based on self-attention mechanism that effectively captures dynamic spatiotemporal correlations of the road network.
#### MT-STGIN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN)
* MT-STGIN, the proposed method.  
---
## EXPERIMENTAL RESULTS { [paper experimental results link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/paper) }
---