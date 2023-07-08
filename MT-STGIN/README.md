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

> In this paper, we directly reproduced the baseline methods (i.e., we do not change training methods and source codes), and these models mainly use grid-search to select hyperparameters, such as GAMN. To be consistent with baselines, the hidden dimension d is set to 64, and each layer’s input and output dimensions have been uploaded to Table 1. We use the grid search for the proposed MT-STGIN model to find the optimal model on the validation dataset because grid search is easy to understand and implement. Especially among all candidate hyperparameter selections, every possibility is tried through loop traversal, and the parameter with the best performance on the validation dataset is the final result. For instance, six hyperparameters are consistent with baseline methods: learning rate of 0.0005, hidden size $d=64$, batch size of 128, epochs of 200, regularization parameter $\lambda=0.001$, and the training method is Adam. In addition, assume our model has four other hyperparameters that need to be defined, i.e., number of blocks $L\in${1, 2, 3, 4, 5}, number of heads $M\in${1, 4, 8, 12}, decay rate {0.3, 0.5, 0.7, 0.9}, and dropout {0.1, 0.3, 0.5, 0.7}. For example, hyperparameter number of blocks has five choices, hyperparameter number of heads has four options, hyperparameter decay rate has four choices, and hyperparameter dropout has four choices, then all hyperparameter combinations have $5 * 4 * 4 * 4$ or 320 kinds. We need to traverse these 320 combinations and find the optimal solution. Note that for these continuous values, sample at equal intervals. In future work, we may try using random search or Bayesian search to find the optimal hyperparameters. 

> In the experiment, the batch size is 128, which divides the training set into 46 iterations in a single epoch. The process of updating the model’s parameters with a batch of data is called one iteration. Second, in the new experiments, we evaluate the prediction model on the validation set after one epoch. If the MAE on the validation set is improved, we update and save the model parameters to replace the last one saved. Finally, when the forecasting performance of the prediction model on the validation set is optimal, the training process ends after many parameter adjustments and experiments. We use an early-stop mechanism in all experiments, and the number of early-stop epochs is set to 50, defined as patience. The early-stop mechanism means the training stops early if the MAE on the validation set is not decreased under the patience before the maximum number of epochs. For instance, parameters are saved when the epoch is 15, and the training ends if the MAE in the validation set is not declined between epochs 15 and 65. To consist with existing studies, we set the target time steps Q and historical time steps P to 6 and 12, respectively, representing the time span is 270 minutes.


> After multiple training steps, the final model framework parameters are determined. Table I presents the number of layers, nodes, output size and related hyperparameters of the MT-STGIN model. We implement the MT-STGIN and baselines in TensorFlow and PyTorch. The server’s one NVIDIA Tesla V100S-PCIE-32GB GPUs and 24 CPU cores are used for model training and testing. Note that the implementation codes of the proposed MT-STGIN model and baseline models are open source, and are available at the personal GitHub homepage [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN).
---
## METRICS

> In order to evaluate the prediction performance of the MT-STGIN model, three metrics are used to determine the difference between the observed values $\rm Y$ and the predicted values $\rm \hat{Y}$ : the root mean square error (RMSE), mean absolute error (MAE), and mean absolute percentage error (MAPE). Note that low MAE, RMSE, and MAPE values indicate a more accurate prediction performance.  

> 1.  MAE (Mean Absolute Error):

$$
{\rm MAE}=\frac{1}{\rm N} \sum_{i=1}^{\rm N}\left|\hat{\rm Y}_{i}-{\rm Y}_{i}\right|
$$

> 2. RMSE (Root Mean Square Error):

$$
{\rm RMSE} =\sqrt{\frac{1}{\rm N} \sum_{i=1}^{\rm N}\left(\hat{\rm Y}_{i}-{\rm Y}_{i}\right)^{2}}
$$

> 3. MAPE (Mean Absolute Percentage Error):

$$
{\rm MAPE}=\frac{100 \%}{\rm N} \sum_{i=1}^{\rm N}\left|\frac{\hat{\rm Y}_{i}-{\rm Y}_{i}}{{\rm Y}_{i}}\right|
$$
  
---

## MT-STGIN AND BASELINES （ALL METHODS' CODES HAVE BEEN REPRODUCED） 
#### HA  [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/ha)  | ARIMA [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/arima)  
#### SVM [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline)

#### LSTM NN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/lstm) | Bi-LSTM NN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/bi_lstm) | FI-RNNs [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/firnn)
#### PSPNN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/pspnn) | MDL [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/mdl)
#### GMAN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/gman) | T-GCN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/tgcn) | AST-GAT [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/astgat) | DCRNN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/dcrnn) | ST-GRAT [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/baseline/st_grat) | MT-STGIN [codes link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN) 
---
## EXPERIMENTAL RESULTS { [paper experimental results link](https://github.com/zouguojian/Traffic-speed-prediction/tree/main/MT-STGIN/paper) }

### PREDICTING PERFORMANCE COMPARISON 
> Performance comparison of different approaches for long-term highway traffic speed prediction  

<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/figs/2.png" width = "1200" height="370"/></div>  
<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/figs/3.png" width = "1200" height="370"/></div>  
<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/figs/4.png" width = "1200" height="370"/></div>  

### INFLUENCE OF EACH COMPONENT

> Performance of the different time steps prediction for distinguished variants  

<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/figs/5.png" width = "1200" height="370"/></div>

### COMPUTATION COST

> Computation cost during the training and inference phases (* means the model train one time on the whole training set) 

<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/figs/6.png" width = "1200" height="370"/></div>

---