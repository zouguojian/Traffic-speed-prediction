# Traffic-flow-prediction

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements.txt文件中指示的包进行安装，才能正常的运行程序！！！</font>
  
>* 首先，使用conda创建一个虚拟环境，如‘conda create traffic_flow’；  
> * 激活环境，conda activate traffic_flow；  
> * 安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0；  
> * 如果安装的是最新的tensorflow环境，也没问题，tensorflow的包按照以下方式进行导入即可：import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()；  
> * 点击 run_train.py文件即可运行代码。
---

## 模型实验结果
### LSTM (1-steps)

|  评价指标     | 6-1 steps|6-3 steps |6-6 steps |
|  ----        | ----     |  ----    |  ----    |
| MAE          | 6.101470 | 6.435726 | 6.985751 |
| RMSE         | 9.306101 | 9.930227 | 11.021970|
| R            | 0.972961 | 0.969166 | 0.961918 |
| R<sup>2</sup>| 0.946399 | 0.938987 | 0.924858 | 
 
### GMAN (1step)  

> Embedding size is 256  

MAE is : 6.119343
RMSE is : 9.470411
R is: 0.972230
R^$2$ is: 0.94452

没加gcn
MAE is : 6.133067
RMSE is : 9.510901
R is: 0.971789
R^$2$ is: 0.944049

### ST-GAT (6-steps)  
#### 1-blocks and 1 heads for spatial, 4-blocks and 1 heads for temporal

|评价指标         |4-6 steps|5-6 steps|6-6 steps|7-6 steps|8-6 steps|9-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
|epoch           |50       |50       |100      |100      |100      |100      |
|embedding       |64       |64       |64       | 64      |64       |64       |
| MAE            |14.322435|6.550879 |6.372804 |6.439330 |6.479654 |6.501210 |
| RMSE           |20.525616|10.132798|9.840441 |9.963966 |10.042691|10.095425|
| R              |0.927124 |0.968368 |0.970164 |0.969012 |0.968415 |0.968201 |
| R<sup>2</sup>  |0.739462 |0.936499 |0.940104 |0.938585 |0.937646 |0.936981 | 

#### 1-blocks and 4 heads for spatial, 1-blocks and 4 heads for temporal

|评价指标         |6-1 steps|6-2 steps|6-3 steps|6-4 steps|6-5 steps|6-6 steps|
|  ----          | ----    |  ----   |  ----   |----     |----     |----     |
|epoch           |100      |100      |100      |100      |100      |100      |
|embedding       |256      |256      |256      | 256     |256      |256      |
| MAE            |5.978483 |6.001878 |6.020515 |6.052269 |6.078716 |6.109353 |
| RMSE           |9.140041 |9.200141 |9.238063 |9.306909 |9.361968 |9.416938 |
| R              |0.974360 |0.974010 |0.973781 |0.973348 |0.973017 |0.972683 |
| R<sup>2</sup>  |0.948295 |0.947623 |0.947196 |0.946412 |0.945793 |0.945149 | 