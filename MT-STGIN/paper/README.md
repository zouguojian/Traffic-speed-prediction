# TRAFFIC-SPEED-PREDICTION

## Long-term highway traffic speed prediction ability: row (a) presents the prediction error of each step in task ETTG; row (b) indicates the performance of each step in task GTG; row (c) reflects the prediction accuracy of each step in task GTET.

<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/paper/figure/1.png"/></div>

## Degree of fit between the observed and predicted traffic speed values. (a) relevant results in task ETTG; (b) in task GTG; (c) in task GTET.

<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/paper/figure/2.png"/></div>

> To better demonstrate the performance of MT-STGIN, we compare it with the other two optimal spatiotemporal baseline models and visualize the fitting results. Figure shows the visualization results of the predictive fit ability over six target time steps, the black line indicates the linear function (as y = x), and the black dots indicate the degree of deviation between the observed and predicted values. MT- STGIN presents a significant fitting performance on ETTG, GTG, and GTET when the speed is higher than 30 km/h; the discrete degree of black dots contained in MT-STGIN is lower than that in the other two models. For example, when the traffic speeds are above 125 km/h, the black dots contained in MT- STGIN is more adjacent to the black line than in the other two state-of-the-art models. The comparison results demonstrate that MT-STGIN has excellent fitting performance for long- term highway traffic speed prediction and may express good application prospects.  
> However, when the traffic speeds are below 30 km/h, the black dots in AST-GAT are closer to the black line than in MT-STGIN. The performance of AST-GAT on low traffic speed prediction may benefit from modeling the different scales of time series, including recent, daily-periodic, and weekly-periodic time series, but not obvious when traffic speeds are above 30 km/h. Therefore, we can incorporate the advantage of AST-GAT into MT-STGIN in future work, improving the prediction performance at low speeds.


## Degree of fit between the observed and predicted traffic speed values. (a) relevant results in task ETTG; (b) in task GTG; (c) in task GTET.

<div align=center><img src ="https://github.com/zouguojian/Traffic-speed-prediction/blob/main/MT-STGIN/paper/figure/3.png"/></div>

> Three road segments are exampled from these three tasks, respectively, and visualized the prediction results for the six- time steps horizon. In the experiment, one hundred continuous samples are randomly sliced from the test set, and the samples’ time interval is 2021.8.13 18:30 to 2021.8.20 00:30. Figure shows that MT-STGIN can accurately fit the changing trend of traffic speed and adapt to complex speed fluctuations, compared with optimal baseline models, GMAN and AST- GAT. For example, the traffic speeds present huge differences between different types of road segments; however, the per- formance of MT-STGIN keeps steady compared with baseline models, such as the samples from 65 to 90 (between 390 and 540 in Figure). MT-STGIN still consistently achieves better results than other baselines, makes predictions close to actual observations, and conquers speed fluctuations. These properties play a vital role in future travel services and traffic control.

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

## EXPERIMENTAL RESULTS
### HA (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.598245 |6.598245 |6.598245 |6.598245 |6.598245 |6.598245 |6.598245 |
| RMSE           |11.027488|11.027488|11.027488|11.027488|11.027488|11.027488|11.027488|
| MAPE           |0.150648 |0.150648 |0.150648 |0.150648 |0.150648 |0.150648 |0.150648 |
| R              |0.923372 |0.923372 |0.923372 |0.923372 |0.923372 |0.923372 |0.923372 |
| R<sup>2</sup>  |0.834505 |0.834505 |0.834505 |0.834505 |0.834505 |0.834505 |0.834505 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.029690 |5.029690 |5.029690 |5.029690 |5.029690 |5.029690 |5.029690 |
| RMSE           |8.455639 |8.455639 |8.455639 |8.455639 |8.455639 |8.455639 |8.455639 |
| MAPE           |0.075200 |0.075200 |0.075200 |0.075200 |0.075200 |0.075200 |0.075200 |
| R              |0.831859 |0.831859 |0.831859 |0.831859 |0.831859 |0.831859 |0.831859 |
| R<sup>2</sup>  |0.688987 |0.688987 |0.688987 |0.688987 |0.688987 |0.688987 |0.688987 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.918259 |7.918259 |7.918259 |7.918259 |7.918259 |7.918259 |7.918259 |
| RMSE           |11.923928|11.923928|11.923928|11.923928|11.923928|11.923928|11.923928|
| MAPE           |0.196434 |0.196434 |0.196434 |0.196434 |0.196434 |0.196434 |0.196434 |
| R              |0.914665 |0.914665 |0.914665 |0.914665 |0.914665 |0.914665 |0.914665 |
| R<sup>2</sup>  |0.836266 |0.836266 |0.836266 |0.836266 |0.836266 |0.836266 |0.836266 |
---
### ARIMA (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.626915 |4.647953 |4.753901 |4.665786 |4.861238 |4.727363 |4.713859 |
| RMSE           |7.632752 |7.821253 |7.788622 |7.538529 |7.901464 |7.512257 |7.700542 |
| MAPE           |0.171705 |0.120875 |0.134842 |0.114128 |0.133091 |0.105187 |0.129971 |
| R              |0.959264 |0.957374 |0.957619 |0.960159 |0.956712 |0.960441 |0.958580 |
| R<sup>2</sup>  |0.920017 |0.916488 |0.916937 |0.921815 |0.915240 |0.922293 |0.918788 |
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.603809 |5.637829 |5.823972 |6.047287 |6.172194 |6.106020 |5.898518 |
| RMSE           |8.914737 |8.824012 |9.261334 |9.508310 |9.488410 |9.289898 |9.218172 |
| MAPE           |0.082443 |0.101620 |0.075173 |0.079390 |0.086643 |0.098172 |0.087241 |
| R              |0.809119 |0.815076 |0.794910 |0.774498 |0.788932 |0.789219 |0.795328 |
| R<sup>2</sup>  |0.651203 |0.663376 |0.629387 |0.592328 |0.620503 |0.618989 |0.629500 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.339334 |7.256662 |7.363538 |7.517883 |7.359343 |7.584464 |7.403537 |
| RMSE           |11.568608|11.343909|11.220085|11.492640|11.125178|11.517343|11.379124|
| MAPE           |0.200755 |0.223617 |0.210570 |0.190724 |0.169315 |0.223585 |0.203094 |
| R              |0.920375 |0.923320 |0.925978 |0.921117 |0.924704 |0.920129 |0.922584 |
| R<sup>2</sup>  |0.846751 |0.852343 |0.857409 |0.848233 |0.854639 |0.846284 |0.850953 |
---
### SVM (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.816173 |5.050724 |4.883361 |5.099447 |5.225978 |5.277595 |5.058880 |
| RMSE           |8.735231 |9.115963 |8.671846 |9.090171 |9.282398 |9.244173 |9.026395 |
| MAPE           |0.118592 |0.138792 |0.108577 |0.181763 |0.129680 |0.144779 |0.137030 |
| R              |0.946558 |0.942514 |0.947510 |0.942772 |0.940343 |0.940733 |0.943363 |
| R<sup>2</sup>  |0.895072 |0.887268 |0.896586 |0.886557 |0.882371 |0.882990 |0.888462 |
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.176205 |5.268431 |5.125441 |5.397734 |5.397268 |5.442187 |5.301211 |
| RMSE           |8.954669 |8.791142 |8.518839 |8.738810 |8.745976 |9.062547 |8.803688 |
| MAPE           |0.072027 |0.078186 |0.090606 |0.081443 |0.103722 |0.073252 |0.083206 |
| R              |0.804310 |0.823605 |0.826811 |0.817597 |0.820665 |0.806513 |0.816483 |
| R<sup>2</sup>  |0.638860 |0.674763 |0.679700 |0.664833 |0.669303 |0.645126 |0.662247 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.257352 |7.130939 |7.340405 |7.641547 |7.502731 |7.485142 |7.393020 |
| RMSE           |11.221017|10.887661|11.333131|11.812502|11.747690|11.383118|11.401835|
| MAPE           |0.179058 |0.158030 |0.212177 |0.201804 |0.228072 |0.209624 |0.198128 |
| R              |0.925146 |0.928372 |0.923125 |0.917215 |0.918493 |0.923988 |0.922675 |
| R<sup>2</sup>  |0.855315 |0.860677 |0.851072 |0.840221 |0.841645 |0.853235 |0.850323 |
---
### LSTM (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.403996 |4.551445 |4.638916 |4.579973 |4.801920 |4.689835 |4.611014 |
| RMSE           |7.405439 |7.718271 |7.662394 |7.441790 |7.873712 |7.487831 |7.600085 |
| MAPE           |0.171577 |0.121994 |0.132865 |0.115042 |0.134420 |0.107697 |0.130599 |
| R              |0.961783 |0.958583 |0.959078 |0.961326 |0.957319 |0.960825 |0.959804 |
| R<sup>2</sup>  |0.924679 |0.918612 |0.919487 |0.923640 |0.915727 |0.922715 |0.920798 |
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.273601 |5.395432 |5.544694 |5.779489 |5.941308 |5.903246 |5.639628 |
| RMSE           |8.496929 |8.686764 |8.971381 |9.258606 |9.304743 |9.068303 |8.969183 |
| MAPE           |0.078137 |0.101390 |0.071518 |0.076657 |0.083822 |0.094253 |0.084296 |
| R              |0.826605 |0.821288 |0.808245 |0.785163 |0.797471 |0.798788 |0.806311 |
| R<sup>2</sup>  |0.683015 |0.674481 |0.653059 |0.614120 |0.635946 |0.637791 |0.649873 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.119856 |7.191312 |7.342861 |7.578125 |7.384203 |7.636884 |7.375540 |
| RMSE           |11.265826|11.297305|11.282042|11.676560|11.217584|11.682571|11.405344|
| MAPE           |0.200004 |0.225019 |0.213546 |0.197244 |0.176159 |0.232828 |0.207467 |
| R              |0.924762 |0.924273 |0.925306 |0.918754 |0.923826 |0.918317 |0.922514 |
| R<sup>2</sup>  |0.854863 |0.853652 |0.856003 |0.843390 |0.852284 |0.841906 |0.850376 |
---
### Bi-LSTM (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.405321 |4.518141 |4.612855 |4.538890 |4.741737 |4.628935 |4.574314 |
| RMSE           |7.404375 |7.675109 |7.682601 |7.455969 |7.857286 |7.442105 |7.588015 |
| MAPE           |0.168112 |0.119241 |0.131140 |0.113328 |0.130864 |0.105835 |0.128087 |
| R              |0.961845 |0.959089 |0.958867 |0.961125 |0.957334 |0.961210 |0.959899 |
| R<sup>2</sup>  |0.924701 |0.919520 |0.919061 |0.923348 |0.916079 |0.923656 |0.921049 |
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.181705 |5.285832 |5.468326 |5.733428 |5.858315 |5.799661 |5.554544 |
| RMSE           |8.403192 |8.574498 |8.921825 |9.203254 |9.205080 |8.935622 |8.878939 |
| MAPE           |0.078004 |0.101452 |0.071523 |0.076626 |0.083166 |0.094057 |0.084138 |
| R              |0.830943 |0.826379 |0.810672 |0.787877 |0.802517 |0.805520 |0.810675 |
| R<sup>2</sup>  |0.689970 |0.682841 |0.656882 |0.618720 |0.643703 |0.648312 |0.656883 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.105305 |7.151662 |7.260796 |7.487024 |7.269364 |7.537749 |7.301984 |
| RMSE           |11.265682|11.281952|11.220137|11.583176|11.075891|11.616089|11.342165|
| MAPE           |0.219980 |0.232877 |0.209893 |0.190719 |0.175507 |0.239693 |0.211445 |
| R              |0.925283 |0.925102 |0.926485 |0.920520 |0.926212 |0.919669 |0.923862 |
| R<sup>2</sup>  |0.854867 |0.854050 |0.857579 |0.845885 |0.855992 |0.843700 |0.852029 |
---
### FI-RNNs (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.398055 |4.545518 |4.635656 |4.565698 |4.769176 |4.657133 |4.595206 |
| RMSE           |7.403328 |7.707429 |7.658887 |7.452439 |7.858686 |7.460417 |7.591969 |
| MAPE           |0.169929 |0.121353 |0.132889 |0.114642 |0.131466 |0.106373 |0.129442 |
| R              |0.961948 |0.958606 |0.958973 |0.961102 |0.957305 |0.960979 |0.959773 |
| R<sup>2</sup>  |0.924722 |0.918841 |0.919560 |0.923421 |0.916049 |0.923280 |0.920967 |
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.266936 |5.421348 |5.636891 |5.774329 |5.920276 |5.856885 |5.646111 |
| RMSE           |8.483817 |8.660330 |9.018441 |9.248565 |9.270723 |9.018530 |8.954746 |
| MAPE           |0.078535 |0.102202 |0.072330 |0.076636 |0.083585 |0.093695 |0.084497 |
| R              |0.827102 |0.823153 |0.807782 |0.785630 |0.799314 |0.802019 |0.807357 |
| R<sup>2</sup>  |0.683993 |0.676460 |0.649410 |0.614956 |0.638603 |0.641756 |0.650999 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.121698 |7.189935 |7.319066 |7.545589 |7.325423 |7.568237 |7.344992 |
| RMSE           |11.287559|11.292755|11.262303|11.653951|11.149487|11.638626|11.382428|
| MAPE           |0.190447 |0.222089 |0.210297 |0.192372 |0.165135 |0.227039 |0.201230 |
| R              |0.924845 |0.924274 |0.925507 |0.919197 |0.924897 |0.918940 |0.922890 |
| R<sup>2</sup>  |0.854303 |0.853770 |0.856507 |0.843996 |0.854072 |0.843093 |0.850977 |
---
### PSPNN (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.353647 |4.436225 |4.534059 |4.488236 |4.715021 |4.571190 |4.516397 |
| RMSE           |7.324924 |7.563156 |7.555393 |7.346906 |7.786712 |7.301580 |7.481787 |
| MAPE           |0.170713 |0.118265 |0.130777 |0.114020 |0.133129 |0.103548 |0.128409 |
| R              |0.962813 |0.960301 |0.960180 |0.962221 |0.958014 |0.962597 |0.960994 |
| R<sup>2</sup>  |0.926308 |0.921851 |0.921720 |0.925574 |0.917579 |0.926512 |0.923244 |
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.018109 |5.132795 |5.258064 |5.508495 |5.584318 |5.481236 |5.330503 |
| RMSE           |8.251502 |8.415823 |8.760375 |9.008966 |8.936558 |8.648383 |8.674441 |
| MAPE           |0.076827 |0.101001 |0.070024 |0.074707 |0.081328 |0.093152 |0.082840 |
| R              |0.838044 |0.834155 |0.818309 |0.797599 |0.815126 |0.818897 |0.820318 |
| R<sup>2</sup>  |0.701062 |0.694471 |0.669188 |0.634648 |0.664187 |0.670559 |0.672506 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.052771 |7.004601 |7.083312 |7.243506 |7.039853 |7.323674 |7.124619 |
| RMSE           |11.129402|11.062735|10.921598|11.247130|10.737179|11.290270|11.066346|
| MAPE           |0.194727 |0.217832 |0.206510 |0.194110 |0.169886 |0.225496 |0.201427 |
| R              |0.926700 |0.927504 |0.930118 |0.924659 |0.930107 |0.923546 |0.927081 |
| R<sup>2</sup>  |0.858357 |0.859667 |0.865057 |0.854697 |0.864665 |0.852345 |0.859138 |
---
### MDL (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.567766 |4.668223 |4.802578 |4.801348 |5.006125 |4.777165 |4.770534 |
| RMSE           |7.633171 |7.941956 |8.049773 |8.048460 |8.376655 |7.766957 |7.972980 |
| MAPE           |0.179061 |0.133763 |0.145827 |0.125346 |0.143087 |0.107506 |0.139098 |
| R              |0.959578 |0.956467 |0.954803 |0.954715 |0.951642 |0.957692 |0.955746 |
| R<sup>2</sup>  |0.919975 |0.913826 |0.911140 |0.910682 |0.904617 |0.916846 |0.912835 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.922186 |5.050283 |5.199652 |5.399734 |5.446237 |5.300889 |5.219830 |
| RMSE           |8.098513 |8.267899 |8.640902 |8.907971 |8.762857 |8.452044 |8.526292 |
| MAPE           |0.075204 |0.097207 |0.069290 |0.073145 |0.079649 |0.089566 |0.080677 |
| R              |0.844471 |0.841278 |0.825104 |0.804379 |0.824432 |0.828323 |0.827975 |
| R<sup>2</sup>  |0.712045 |0.705117 |0.678149 |0.642794 |0.677114 |0.685347 |0.683597 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.115840 |7.157158 |7.221766 |7.438228 |7.307103 |7.557099 |7.299532 |
| RMSE           |11.168698|11.162749|11.012158|11.347979|10.966957|11.412749|11.179714|
| MAPE           |0.191805 |0.220924 |0.201937 |0.187723 |0.168965 |0.218110 |0.198244 |
| R              |0.926191 |0.926146 |0.928884 |0.923251 |0.927107 |0.921720 |0.925516 |
| R<sup>2</sup>  |0.857355 |0.857118 |0.862810 |0.852080 |0.858811 |0.849124 |0.856238 |
---
### T-GCN (Multi-steps)  
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.867914 |4.915519 |4.991863 |4.900115 |5.071895 |4.880197 |4.937917 |
| RMSE           |8.007648 |8.129066 |8.126287 |7.989766 |8.363378 |7.866967 |8.081999 |
| MAPE           |0.183295 |0.130065 |0.138732 |0.119302 |0.142765 |0.108532 |0.137115 |
| R              |0.955008 |0.953810 |0.953676 |0.955036 |0.951288 |0.956496 |0.954193 |
| R<sup>2</sup>  |0.911931 |0.909718 |0.909443 |0.911980 |0.904920 |0.914690 |0.910435 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.881372 |4.909656 |4.990881 |5.137515 |5.178477 |5.040731 |5.023106 |
| RMSE           |8.011009 |8.144530 |8.430680 |8.608794 |8.550889 |8.144875 |8.318192 |
| MAPE           |0.074088 |0.096569 |0.066887 |0.069687 |0.075928 |0.086002 |0.078194 |
| R              |0.848566 |0.846399 |0.834181 |0.818277 |0.833650 |0.842130 |0.837184 |
| R<sup>2</sup>  |0.718234 |0.713851 |0.693619 |0.666385 |0.692546 |0.707803 |0.698853 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.292850 |7.264247 |7.260326 |7.410805 |7.263291 |7.412676 |7.317366 |
| RMSE           |11.339722|11.330866|11.029550|11.286448|10.923553|11.250577|11.194583|
| MAPE           |0.192573 |0.218989 |0.201144 |0.184970 |0.164272 |0.216086 |0.196339 |
| R              |0.923681 |0.923698 |0.928714 |0.923975 |0.927447 |0.923888 |0.925214 |
| R<sup>2</sup>  |0.852953 |0.852782 |0.862376 |0.853680 |0.859926 |0.853382 |0.855855 |
---
### DCRNN (Multi-steps)  
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.730402 |4.711142 |4.700361 |4.727126 |4.761065 |4.794417 |4.737419 |
| RMSE           |7.731863 |7.739123 |7.725201 |7.759669 |7.777472 |7.858567 |7.765448 |
| MAPE           |0.131417 |0.130804 |0.133867 |0.135671 |0.134084 |0.136295 |0.133690 |
| R              |0.958374 |0.958418 |0.958465 |0.958115 |0.957916 |0.957054 |0.958055 |
| R<sup>2</sup>  |0.918145 |0.917976 |0.918259 |0.917521 |0.917129 |0.915377 |0.917402 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |5.218989 |5.141128 |5.105221 |5.100863 |5.099192 |5.116236 |5.130271 |
| RMSE           |8.667619 |8.588108 |8.559472 |8.542141 |8.536303 |8.546819 |8.573530 |
| MAPE           |0.082794 |0.081971 |0.081556 |0.080885 |0.080372 |0.080445 |0.081337 |
| R              |0.821436 |0.825535 |0.827023 |0.827796 |0.827869 |0.827422 |0.826100 |
| R<sup>2</sup>  |0.672513 |0.678452 |0.680566 |0.681894 |0.682355 |0.681604 |0.679564 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |7.253874 |7.323273 |7.389776 |7.408976 |7.444117 |7.467514 |7.381255 |
| RMSE           |11.236532|11.301529|11.361451|11.427432|11.435319|11.478284|11.373731|
| MAPE           |0.226070 |0.215237 |0.212807 |0.214020 |0.212680 |0.215993 |0.216135 |
| R              |0.924863 |0.924069 |0.923179 |0.922356 |0.922192 |0.921665 |0.923050 |
| R<sup>2</sup>  |0.854641 |0.852971 |0.851432 |0.849710 |0.849518 |0.848396 |0.851111 |
---
### AST-GAT (Multi-steps)
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.497184 |4.613954 |4.649611 |4.607247 |4.772495 |4.547935 |4.614738 |
| RMSE           |7.535124 |7.784701 |7.653306 |7.662314 |7.960739 |7.460128 |7.677789 |
| MAPE           |0.168395 |0.120126 |0.129663 |0.110013 |0.129681 |0.100302 |0.126363 |
| R              |0.960744 |0.957982 |0.959282 |0.958993 |0.956530 |0.961085 |0.959074 |
| R<sup>2</sup>  |0.922018 |0.917205 |0.919678 |0.919047 |0.913854 |0.923286 |0.919170 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.679883 |4.652782 |4.718185 |4.913333 |4.889099 |4.672708 |4.754332 |
| RMSE           |7.822722 |7.935115 |8.138360 |8.359120 |8.272555 |7.832850 |8.062853 |
| MAPE           |0.070963 |0.093704 |0.063064 |0.066619 |0.072517 |0.082804 |0.074945 |
| R              |0.855312 |0.853464 |0.845293 |0.828927 |0.844036 |0.854280 |0.846849 |
| R<sup>2</sup>  |0.731323 |0.728377 |0.714497 |0.685455 |0.712236 |0.729761 |0.717058 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.892128 |6.818961 |6.961788 |6.997957 |6.797758 |7.006877 |6.912578 |
| RMSE           |10.922516|10.779595|10.728664|10.848628|10.346283|10.841717|10.746211|
| MAPE           |0.201657 |0.221000 |0.211088 |0.187143 |0.171407 |0.221291 |0.202264 |
| R              |0.929379 |0.931107 |0.932696 |0.929962 |0.935144 |0.929522 |0.931275 |
| R<sup>2</sup>  |0.863574 |0.866758 |0.869783 |0.864811 |0.874340 |0.863845 |0.867170 |
---
### GMAN (Multi-steps)  
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.334882 |4.313689 |4.360220 |4.222983 |4.436746 |4.213836 |4.313725 |
| RMSE           |7.332422 |7.419843 |7.373770 |7.132550 |7.515837 |7.016173 |7.300442 |
| MAPE           |0.172884 |0.114356 |0.129397 |0.111522 |0.130163 |0.095924 |0.125708 |
| R              |0.962519 |0.961689 |0.962053 |0.964335 |0.960933 |0.965496 |0.962822 |
| R<sup>2</sup>  |0.926157 |0.924784 |0.925438 |0.929854 |0.923214 |0.932145 |0.926920 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.740033 |4.676142 |4.746842 |4.874263 |4.917791 |4.727566 |4.780439 |
| RMSE           |7.906706 |8.000947 |8.240516 |8.415353 |8.332593 |7.890183 |8.133700 |
| MAPE           |0.072855 |0.095317 |0.063751 |0.066442 |0.072201 |0.081023 |0.075265 |
| R              |0.852234 |0.851333 |0.841432 |0.827349 |0.842167 |0.852458 |0.844447 |
| R<sup>2</sup>  |0.725523 |0.723852 |0.707285 |0.681209 |0.708044 |0.725791 |0.712064 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.999796 |6.907157 |6.978471 |7.049780 |6.833681 |7.030167 |6.966509 |
| RMSE           |11.002648|10.892133|10.725927|10.940084|10.504736|10.907470|10.830129|
| MAPE           |0.204040 |0.216095 |0.206211 |0.186343 |0.165788 |0.218526 |0.199500 |
| R              |0.928246 |0.929650 |0.932680 |0.928755 |0.933065 |0.928757 |0.930161 |
| R<sup>2</sup>  |0.861565 |0.863961 |0.869849 |0.862523 |0.870461 |0.862188 |0.865088 |
---
### ST-GRAT (Multi-steps)  
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.234032 |4.306022 |4.393604 |4.495383 |4.588520 |4.675550 |4.448852 |
| RMSE           |7.280299 |7.386465 |7.502555 |7.658772 |7.788842 |7.913568 |7.591644 |
| MAPE           |0.124619 |0.127905 |0.131194 |0.132167 |0.135779 |0.135204 |0.131145 |
| R              |0.963028 |0.962207 |0.961345 |0.960235 |0.959324 |0.958469 |0.960578 |
| R<sup>2</sup>  |0.927299 |0.925152 |0.922765 |0.919506 |0.916730 |0.914032 |0.920915 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.772299 |4.871571 |4.995002 |5.165320 |5.325081 |5.484719 |5.102332 |
| RMSE           |8.264096 |8.362779 |8.491458 |8.677082 |8.858995 |9.038960 |8.619865 |
| MAPE           |0.076821 |0.078492 |0.080265 |0.082695 |0.084784 |0.086787 |0.081640 |
| R              |0.839164 |0.836642 |0.833662 |0.829368 |0.824790 |0.820529 |0.829394 |
| R<sup>2</sup>  |0.702527 |0.695305 |0.685856 |0.671988 |0.658102 |0.644035 |0.676303 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.771106 |6.891242 |6.995427 |7.106168 |7.192948 |7.274607 |7.038582 |
| RMSE           |10.776956|10.918130|11.055330|11.224825|11.347829|11.465987|11.134088|
| MAPE           |0.185265 |0.189862 |0.193490 |0.197077 |0.200330 |0.201765 |0.194631 |
| R              |0.931098 |0.929745 |0.928495 |0.927014 |0.926030 |0.925127 |0.927698 |
| R<sup>2</sup>  |0.866405 |0.862899 |0.859438 |0.855106 |0.851929 |0.848844 |0.857436 |
---
### MT-STGIN (Multi-steps) 
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.167658 |4.152724 |4.225301 |4.129288 |4.343058 |4.132023 |4.191676 |
| RMSE           |7.233356 |7.331061 |7.390471 |7.154604 |7.542089 |6.995177 |7.276548 |
| MAPE           |0.161414 |0.109230 |0.118732 |0.107747 |0.125871 |0.093290 |0.119381 |
| R              |0.963540 |0.962672 |0.961923 |0.964170 |0.960655 |0.965742 |0.963101 |
| R<sup>2</sup>  |0.928139 |0.926573 |0.925100 |0.929420 |0.922677 |0.932550 |0.927397 |  
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.682971 |4.588183 |4.672132 |4.860928 |4.832805 |4.613521 |4.708423 |
| RMSE           |7.956369 |8.014418 |8.377290 |8.592709 |8.402509 |7.933352 |8.216736 |
| MAPE           |0.072753 |0.096213 |0.064088 |0.067078 |0.072708 |0.082680 |0.075920 |
| R              |0.850528 |0.851597 |0.836524 |0.820580 |0.839457 |0.851166 |0.841604 |
| R<sup>2</sup>  |0.722064 |0.722921 |0.697487 |0.667630 |0.703124 |0.722782 |0.706155 | 
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.884369 |6.772877 |6.750407 |6.943268 |6.745666 |6.883471 |6.830010 |
| RMSE           |10.983397|10.887800|10.592183|10.859518|10.464038|10.828980|10.770837|
| MAPE           |0.183126 |0.204971 |0.188529 |0.169706 |0.149622 |0.202187 |0.183024 |
| R              |0.928699 |0.930042 |0.934506 |0.930023 |0.933884 |0.929958 |0.931160 |
| R<sup>2</sup>  |0.862049 |0.864070 |0.873075 |0.864540 |0.871463 |0.864164 |0.866561 | 
---
### MT-STGIN-1 (Multi-steps) does not consider semantic enhancement and uses a fully connected layer instead of it
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.177642 |4.176058 |4.251300 |4.145226 |4.376534 |4.163964 |4.215121 |
| RMSE           |7.228418 |7.361589 |7.406847 |7.139518 |7.551705 |7.017263 |7.286373 |
| MAPE           |0.160313 |0.110279 |0.116948 |0.107044 |0.125626 |0.092810 |0.118837 |
| R              |0.963708 |0.962417 |0.961815 |0.964376 |0.960607 |0.965560 |0.963063 |
| R<sup>2</sup>  |0.928237 |0.925961 |0.924767 |0.929717 |0.922479 |0.932124 |0.927201 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.684889 |4.609807 |4.688076 |4.875859 |4.847807 |4.591308 |4.716291 |
| RMSE           |7.968931 |8.013228 |8.370390 |8.584761 |8.387224 |7.914838 |8.210434 |
| MAPE           |0.072765 |0.096269 |0.064185 |0.067153 |0.072686 |0.082481 |0.075923 |
| R              |0.849922 |0.851407 |0.836587 |0.820695 |0.839828 |0.851646 |0.841635 |
| R<sup>2</sup>  |0.721186 |0.723003 |0.697986 |0.668245 |0.704203 |0.724075 |0.706605 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.893314 |6.773089 |6.744419 |6.944394 |6.757647 |6.877660 |6.831753 |
| RMSE           |10.997348|10.866569|10.597647|10.884158|10.464911|10.824667|10.774096|
| MAPE           |0.182854 |0.205598 |0.186801 |0.170593 |0.148744 |0.199221 |0.182302 |
| R              |0.928560 |0.930389 |0.934473 |0.929745 |0.933925 |0.930038 |0.931164 |
| R<sup>2</sup>  |0.861698 |0.864599 |0.872944 |0.863924 |0.871442 |0.864273 |0.866480 |
---
### MT-STGIN-2 (Multi-steps) does not consider the physical relationship in the highway network
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.210468 |4.184520 |4.263428 |4.163926 |4.383785 |4.180335 |4.231077 |
| RMSE           |7.275176 |7.355039 |7.403964 |7.176252 |7.572408 |7.016400 |7.301978 |
| MAPE           |0.162226 |0.110566 |0.117515 |0.107430 |0.127172 |0.094586 |0.119916 |
| R              |0.963280 |0.962508 |0.961859 |0.963999 |0.960394 |0.965565 |0.962916 |
| R<sup>2</sup>  |0.927306 |0.926092 |0.924826 |0.928992 |0.922054 |0.932140 |0.926889 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.709673 |4.618034 |4.704633 |4.883819 |4.873705 |4.614575 |4.734074 |
| RMSE           |8.015878 |8.043723 |8.419726 |8.612580 |8.451103 |7.944235 |8.251829 |
| MAPE           |0.072966 |0.097052 |0.064694 |0.067436 |0.073436 |0.083138 |0.076454 |
| R              |0.848864 |0.851445 |0.835677 |0.820587 |0.838159 |0.851574 |0.841006 |
| R<sup>2</sup>  |0.717891 |0.720891 |0.694415 |0.666091 |0.699680 |0.722021 |0.703639 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.888876 |6.795491 |6.747041 |6.946672 |6.745324 |6.910945 |6.839058 |
| RMSE           |10.971483|10.922400|10.636802|10.861975|10.452403|10.880205|10.789099|
| MAPE           |0.182843 |0.207045 |0.188223 |0.169515 |0.149821 |0.201842 |0.183215 |
| R              |0.929007 |0.929760 |0.934032 |0.930073 |0.934126 |0.929399 |0.931041 |
| R<sup>2</sup>  |0.862348 |0.863204 |0.872003 |0.864479 |0.871749 |0.862876 |0.866108 |
---
### MT-STGIN-3 (Multi-steps) does not consider the physical relationship in the highway network
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.183303 |4.160127 |4.242272 |4.140976 |4.366398 |4.149156 |4.207039 |
| RMSE           |7.234445 |7.318758 |7.381603 |7.151853 |7.537702 |6.984626 |7.270264 |
| MAPE           |0.161419 |0.112162 |0.118946 |0.109858 |0.128547 |0.093780 |0.120785 |
| R              |0.963665 |0.962867 |0.962088 |0.964233 |0.960729 |0.965871 |0.963220 |
| R<sup>2</sup>  |0.928117 |0.926820 |0.925279 |0.929474 |0.922767 |0.932754 |0.927523 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.719758 |4.602948 |4.690462 |4.893186 |4.857397 |4.605904 |4.728276 |
| RMSE           |7.980271 |7.987528 |8.362018 |8.576801 |8.389899 |7.913639 |8.205550 |
| MAPE           |0.072897 |0.096254 |0.064279 |0.067254 |0.072995 |0.083027 |0.076117 |
| R              |0.848993 |0.851892 |0.836403 |0.820224 |0.839343 |0.851284 |0.841320 |
| R<sup>2</sup>  |0.720392 |0.724777 |0.698589 |0.668860 |0.704014 |0.724158 |0.706954 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.892833 |6.791966 |6.759236 |6.951820 |6.757033 |6.885289 |6.839696 |
| RMSE           |10.980077|10.881515|10.627948|10.873178|10.468319|10.825764|10.777535|
| MAPE           |0.184003 |0.204968 |0.187040 |0.169737 |0.149705 |0.200275 |0.182621 |
| R              |0.928680 |0.864227 |0.933986 |0.929748 |0.933704 |0.929867 |0.930970 |
| R<sup>2</sup>  |0.862132 |0.863204 |0.872216 |0.871358 |0.871749 |0.864245 |0.866395 |
---
### MT-STGIN-4 (Multi-steps) does not consider error propagation in the long-term prediction stage and uses dynamic decoding
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.269177 |4.255263 |4.330823 |4.232817 |4.445578 |4.240977 |4.295773 |
| RMSE           |7.277377 |7.453777 |7.443018 |7.183141 |7.643382 |7.080325 |7.349233 |
| MAPE           |0.163595 |0.112126 |0.120036 |0.110333 |0.127711 |0.094917 |0.121453 |
| R              |0.963308 |0.961557 |0.961570 |0.964093 |0.959810 |0.965060 |0.962545 |
| R<sup>2</sup>  |0.927262 |0.924095 |0.924031 |0.928855 |0.920586 |0.930898 |0.925940 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.657037 |4.606904 |4.724390 |4.904016 |4.874632 |4.642570 |4.734925 |
| RMSE           |7.947700 |7.985073 |8.429362 |8.646348 |8.433814 |7.995177 |8.244118 |
| MAPE           |0.073029 |0.096027 |0.065073 |0.067592 |0.073453 |0.083490 |0.076444 |
| R              |0.851478 |0.853481 |0.834795 |0.818742 |0.839001 |0.849358 |0.841105 |
| R<sup>2</sup>  |0.722670 |0.724946 |0.693715 |0.663468 |0.700908 |0.718445 |0.704193 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.944869 |6.924512 |6.940500 |7.088108 |6.896299 |7.067578 |6.976977 |
| RMSE           |11.082523|11.018495|10.777887|11.025302|10.603253|11.055566|10.928588|
| MAPE           |0.185631 |0.218505 |0.194530 |0.179919 |0.155483 |0.206151 |0.190036 |
| R              |0.927386 |0.928301 |0.932122 |0.927786 |0.932137 |0.926865 |0.929067 |
| R<sup>2</sup>  |0.859548 |0.860787 |0.868585 |0.860372 |0.868020 |0.858421 |0.862624 |
---
### MT-STGIN-5 (Multi-steps) removes the multi-task learning component, using two fully connected layers instead of it
#### toll to gantry 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.175243 |4.145744 |4.241208 |4.138994 |4.365584 |4.135049 |4.200304 |
| RMSE           |7.234051 |7.298687 |7.378065 |7.148160 |7.545243 |6.954005 |7.262039 |
| MAPE           |0.162539 |0.110751 |0.119312 |0.109043 |0.127833 |0.094084 |0.120594 |
| R              |0.963647 |0.963087 |0.962132 |0.964320 |0.960712 |0.966183 |0.963330 |
| R<sup>2</sup>  |0.928125 |0.927221 |0.925351 |0.929547 |0.922612 |0.933342 |0.927687 | 
#### gantry to gantry
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |4.691376 |4.601727 |4.675421 |4.870234 |4.842151 |4.608901 |4.714968 |
| RMSE           |7.965910 |8.009786 |8.371257 |8.583561 |8.382399 |7.918375 |8.209071 |
| MAPE           |0.072833 |0.096339 |0.064138 |0.067229 |0.072876 |0.083071 |0.076081 |
| R              |0.850104 |0.851950 |0.836805 |0.820612 |0.840270 |0.851698 |0.841868 |
| R<sup>2</sup>  |0.721397 |0.723241 |0.697923 |0.668338 |0.704543 |0.723828 |0.706703 |
#### gantry to toll 
|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|total avg|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |----     |
| MAE            |6.936972 |6.797699 |6.768142 |6.953649 |6.768620 |6.906447 |6.855255 |
| RMSE           |11.000627|10.909314|10.639629|10.890698|10.481229|10.859960|10.798386|
| MAPE           |0.184269 |0.210588 |0.191995 |0.174961 |0.152705 |0.204985 |0.186584 |
| R              |0.928562 |0.929938 |0.933998 |0.929696 |0.933763 |0.929606 |0.930901 |
| R<sup>2</sup>  |0.861616 |0.863532 |0.871935 |0.863761 |0.871041 |0.863386 |0.865877 |
---