# Purpose of the new study

The Java dataset gathered in this work is reused in the new study: Thinking Like a Developer? Comparing the Attention of Humans with Neural Models of Code.

## Usage of the Model
We are interested in the attention weights produced by the neural model during inference, thus we use the model for inference and for train

## Computing Environment

We used the following setup:
- Operating System: Ubuntu 18.04.5 LTS
- Kernel: Linux 4.15.0-136-generic
- Architecture: x86-64
- conda 4.9.2
- x2 GPUs: Tesla T4 with 16 GB each
- RAM: 252 GB
- Nvidia driver (as reported by nvidia-smi command): NVIDIA-SMI 450.102.04   Driver Version: 450.102.04   CUDA Version: 11.0
- Nvidia drivers in folder: /usr/local/cuda-10.2

## Python Dependencies

Check the **requirements_conda.txt** file that includes all the packages that we used for our study. We dumped them with: ```conda list -e > requirements_conda.txt```. To create a new conda environment to reproduce our setup run this from the project folder:
```console
conda create --name condaenv --file requirements_conda.txt
```
where **condaenv** is an arbitrary name for the folder with your dependencies. Make sure to have the two channels (conda-forge and anaconda) by checking ```conda config --show channels```. If you do not have them run the following:
- ```conda config --append channels conda-forge```
-  ```conda config --append channels anaconda```.

## Folder management
we assume that you downloaded the data from the original paper in the folder **dataset_convolutional-attention** as a sibling of the folder containing this project.

## Training Details
We use the model **ConvolutionalCopyAttentionalRecurrentLearner** in the file **copy_conv_rec_learner.py**. We keep the same train and test split and all the training hyperparameters unchanged.

### Main files
To run the analysis we use the two files in the main project folder:
- **manual_copy_conv_rec_learner.py**: to train and test on a single project
- **automatic_copy_conv_rec_learner.py**: to train and test on all the ten projects
- **automatic_copy_conv_rec_visualizer.py**: to create a json containing all the attention weight for the prediction on the test set for all ten projects.

For all of them we have to make sure they have execution rights:
```console
chmod +x manual_copy_conv_rec_learner.py
```

For all of them modify the first line with the your absolute path:
```console
#!/home/<absolute_path>/convolutional-attention/condaenv/bin/python2.7
```

### Manual Run: Single Project

To train the model (**without testing**):
```console
THEANO_FLAGS="optimizer_excluding=conv_dnn" python2 manual_copy_conv_rec_learner.py ../dataset_convolutional-attention/json/libgdx_train_methodnaming.json 10 128
```
To train and **test** the model:
```console
THEANO_FLAGS="optimizer_excluding=conv_dnn" python2 manual_copy_conv_rec_learner.py ../dataset_convolutional-attention/json/libgdx_train_methodnaming.json 1000 128 ../dataset_convolutional-attention/json/libgdx_test_methodnaming.json
```

## Automatic Run: Ten Projects

To run the train and test on all ten projects automatically:
1. export theano flags to disable cudnn (since it doesn't work with my setup)
    ```console
    export THEANO_FLAGS="optimizer_excluding=conv_dnn"
    ```
1. start the background process to train all the ten models (note that the two path to the datasets will be overridden at runtime by the python script):
    ```console
    nohup ./automatic_copy_conv_rec_learner.py ../dataset_convolutional-attention/json/libgdx_train_methodnaming.json 1000 128 ../dataset_convolutional-attention/json/libgdx_test_methodnaming.json &
    ```

The run will be in the background and it will create log files and trained models in the main project folder.

## Extract Attention Weights: 10 projects
To extract the attention for all ten projects automatically:
1. export theano flags to disable cudnn (since it doesn't work with my setup)
    ```console
    export THEANO_FLAGS="optimizer_excluding=conv_dnn"
    ```
1. start the background process to extract the attention all the ten models:
    ```console
    nohup ./automatic_copy_conv_rec_visualizer.py &
    ```
The run will create files in the main project folder with the prefix GPU that contains the attention weights for every predicted token and for all methods in the relative testing dataset of that project. It will also create logs starting with *visualizer* located in the main project folder.


# Reproducibility Review
The project is well-organized and the files autodocumented. The major difficulty in the project is to set up the dependencies since in 2021 it was difficult to find some of the dependencies from 2016.

## Problem Troubleshooting

1. Missing **ExperimentLogger.py** object.
    When running the models thy try to log their results with this object that is not present, therefore we created a very simple version of it that prints to the screen few important informations to log.
    ```python
    class ExperimentLogger(object):

        def __init__(self, model_name, params):
            self.model_name = model_name
            self.params = params

        def __enter__(self):
            return self

        def __exit__(self, a, b, c):
            return

        def record_results(self, dict_metrics):
            print "Model name:"
            print self.model_name
            print "Model parameters:"
            print self.params
            print "Performance Metrics:"
            print dict_metrics
    ```

1. Cuda drivers not compatible with cudnn required by the project. The solution was to run the train with cudnn option switched off.
    ```console
    THEANO_FLAGS="optimizer_excluding=conv_dnn" python2 copy_conv_rec_learner.py etc...
    ```




