#### This is the code for the article called "AENet: An Asymmetric Encoder Network with Information Enhancement for Change Detection"

##### 1、Run Environment :
The code has been tested on Python 3.5 and 3.7, and the installed version of PyTorch is CUDA 11.3. Other toolkits can be downloaded one by one using pip when errors are occurring during the run.

##### 2、Dataset download and Settings:

The specific sources of three open-source datasets can be found through the citations in the paper. Below are the datasets that I have segmented.

Baidu Cloud Drive: https://pan.baidu.com/s/1rV-ZZYZbcVg6NxoffLxK3Q?pwd=6666

After downloading and splitting the dataset, change the path of your own dataset in the **constants.py** file.

##### 3、Run:
Taking the gz-cd dataset as an example, first switch to the path /AENet/src/:

Training: 

```python
python train.py train --exp_config ../configs/gzcd/config_gzcd_clnet.yaml   
```

Test: 

```python
python train.py eval --exp_config ../configs/gzcd/config_gzcd_clnet.yaml --resume ../exp/gzcd/weights/model_best_clnet.pth --save_on --subset test 
```

Continuing the training from the one breakpoint: 

```
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

 (where PATH_TO_CONFIG_FILE is the configuration in the configs directory, and PATH_TO_CHECKPOINT is the model parameter saved in the exp directory)


The following is the internal structure diagram of the encoding and decoding modules in the paper：

![image](https://github.com/user-attachments/assets/c1f5d05a-40e8-4761-a945-021578b64558)


Declaration: The implementation of the code in this article uses the shell of CDLab.  My code has been highly streamlined based on this.  If further understand and modification of the model's shell are required, please refer to the source code https://github.com/Bobholamovic/CDLab.

