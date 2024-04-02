# Traffic-speed-prediction

## WHAT SHOULD WE PAY ATTENTION TO FOCUS ON THE RUNNING ENVIRONMENT?

<font face="微软雅黑" >Note that we need to install the right packages to guarantee the model runs according to the file requirements.txt！！！</font>
  
>* first, please use the conda tool to create a virtual environment, such as ‘conda create traffic speed’;  
> * second, active the environment, and conda activate traffic speed;   
> * third, build environment, the required environments have been added in the file named requirements.txt; you can use conda as an auxiliary tool to install or pip, e.g., conda install tensorflow==1.13.1;    
> * if you have installed the last TensorFlow version, it’s okay; import tensorflow.compat.v1 as tf and tf.disable_v2_behavior();    
> * finally, please click the run_train.py file; the codes are then running;  
> * Note that our TensorFlow version is 1.14.1 and can also be operated on the 2.0. version.  
---

# HOW TO RUN THE MODEL?

train model on Ningxia-YC or METR-LA:
```
python run_train.py and input 1 to run
```
test model on Ningxia-YC or METR-LA:
```
 python run_train.py and input 0 to test
```


# DATASETS

> For the public dataset, we should better to transfer data type from .h5 to .csv if the resource data type is h5.


# IMPORTANT REFERENCES

If you find this repository useful in your research, please cite the following paper:
```
@article{zou2023will,
  title={When Will We Arrive? A Novel Multi-Task Spatio-Temporal Attention Network Based on Individual Preference for Estimating Travel Time},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Tu, Meiting and Fan, Jing and Li, Ye},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}

@article{zou2023novel,
  title={A novel spatio-temporal generative inference network for predicting the long-term highway traffic speed},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Li, Ye and Wang, Ting},
  journal={Transportation research part C: emerging technologies},
  volume={154},
  pages={104263},
  year={2023},
  publisher={Elsevier}
}

@article{zou2024multi,
  title={Multi-task-based spatiotemporal generative inference network: A novel framework for predicting the highway traffic speed},
  author={Zou, Guojian and Lai, Ziliang and Wang, Ting and Liu, Zongshi and Bao, Jingjue and Ma, Changxi and Li, Ye and Fan, Jing},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121548},
  year={2024},
  publisher={Elsevier}
}
```