# Addendum
This is an **unorganized** list of points that the author of the new study listed as possible fixes to the setup of the theano/cuda environment. Follow the README_NEW_STUDY.md for a clear procedure.

## Learners/ Models available

1. Baseline (TF-IDF)
1. Convolutional Attention
    1. conv_attentional_learner -> Convolutional**Attentional**Model
    1. conv_att_rec_learner (Recurrent) -> ConvolutionalAttentional**Recurrent**Model
1. Copy mechanism (additive)
    1. copy_learner -> **Copy**ConvolutionalAttentionalModel
    1. copy_conv_rec_learner (Recurrent) -> **Copy**Convolutional**Recurrent**AttentionalModel


## Trained model

ConvolutionalCopyAttentionalRecurrent**Learner** -> **Copy**Convolutional**Recurrent**AttentionalModel

# Instruction

## System Admin

1. install python 2.7 libraries for dev
    ```
    sudo apt-get install python2.7-dev
    ```
1. install conda (since it is the only supported virtual environment for Theano and GPU) (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

1. libblas (for usage of fast algebra operations)  (source https://csantill.github.io/RPerformanceWBLAS/)
    ```
    sudo apt-get install libopenblas-base
    ```


## Setup environment

1. pip install numpy
1. pip install scipy
1. pip install theano
1. conda install -c mila-udem/label/pre pygpu=0.7.2
1. conda install -c anaconda cudnn (version between 5>= and 7<=)
1. conda uninstall cudnn



## Setup configuration file

Modify this file **vim /home/<your_user>/.theanorc**

```
[global]
device = cuda
floatx = float32

[cuda]
root = /usr/local/cuda-11.1
```

## Error Debugging

### PyGPU
When running the main script (theano 0.9.0 + cudatoolkit and cudnn as described here)
https://yann-leguilly.gitlab.io/post/2019-10-08-tensorflow-and-cuda/
```
ERROR (theano.gpuarray): pygpu was configured but could not be imported or is too old (version 0.6 or higher required)
```
Solutions discussed here https://github.com/Theano/libgpuarray/issues/514


## Probable Solution

1. .theanorc points at cuda-10.1
1. conda install cudnn=7.6.0=cuda10.0_0
1. conda install theano 0.8.2-py27_0
1. conda install pygpu=0.6.9
1. conda install theano 0.9.0
1. conda install pygpu=0.7.0 libgpuarray=0.7.0 theano=1.0.1=py27_1
    ```
    The following packages will be UPDATED:

    libgpuarray                                       0.6.9-0 --> 0.7.0-0
    pygpu                                   0.6.9-np112py27_0 --> 0.7.0-np112py27_0
    theano                                       0.9.0-py27_1 --> 1.0.1-py27_1
    ```
1. conda install cudatoolkit=10.2
1. export CUDA_ROOT=/usr/local/cuda-10.2
1. conda install cudnn=7.0.5

## RUN WITHOUT CuDNN

### TRAIN Only
THEANO_FLAGS="optimizer_excluding=conv_dnn" python2 copy_conv_rec_learner.py ../dataset_convolutional-attention/json/libgdx_train_methodnaming.json 10 128

### With TEST
THEANO_FLAGS="optimizer_excluding=conv_dnn" python2 copy_conv_rec_learner.py ../dataset_convolutional-attention/json/libgdx_train_methodnaming.json 1000 128 ../dataset_convolutional-attention/json/libgdx_test_methodnaming.json

1. Make the file executable
    chmod +x copy_conv_rec_learner.py
1. insert the header on which python to use
    #!/home/<your_user>/projects/AllamanisCodeSummarization/convolutional-attention/condaenv/bin/python2.7
1. export theano flags
    export THEANO_FLAGS="optimizer_excluding=conv_dnn"
1. run it with nohup
    nohup ./copy_conv_rec_learner.py ../dataset_convolutional-attention/json/libgdx_train_methodnaming.json 1000 128 ../dataset_convolutional-attention/json/libgdx_test_methodnaming.json &



## Add to theanorc for dnn path
[dnn]

include_path=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include
library_path=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64

## Remeber

start script with python2 prefix

# Possible fix

1. run
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/
```

1. create the softlinks as here (https://stackoverflow.com/a/64837555)
```
$ sudo ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublas.so /usr/local/cuda-10.1/lib64/libcublas.so
$ sudo ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublas.so.10 /usr/local/cuda-10.1/lib64/libcublas.so.10
$ sudo ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublasLt.so.10 /usr/local/cuda-10.1/lib64/libcublasLt.so.10
```

1. Run
```
cd /home/<your_user>/projects/AllamanisCodeSummarization/convolutional-attention
conda activate /home/<your_user>/projects/AllamanisCodeSummarization/convolutional-attention/condaenv
THEANO_FLAGS=device=cuda0 python tutorial.py
```


## Reverse engineering CUDA path in Theano

Source of info: https://github.com/Theano/Theano/blob/master/theano/configdefaults.py#L245
Latest Theano Release Github
**cuda.root** (the folder in those listed in the $path that contains nvcc command )
Use this command to find out which folder:
```
for p in $PATH ; do      echo "$p"; ls $p | grep nvcc; done
```
On donkey this is the folder /usr/bin/ that contains nvcc
nvcc then is an executable file that runs:
```
#!/bin/sh

exec /usr/lib/nvidia-cuda-toolkit/bin/nvcc "$@"
```

Source of info: my current Theano installation 0.9.0
But if I check in Theano itself my path is the following:
```python
import theano
print theano.config.cuda.root
```
```
/usr/local/cuda-10.2
```
