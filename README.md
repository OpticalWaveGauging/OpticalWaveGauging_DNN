## About

Data and code to implement Buscombe et al (2019) optical wave gauging (OWG) using deep neural networks, detailed in the paper:

> Buscombe, Carini, Harrison, Chickadel, and Warrick (in review) Optical wave gauging with deep neural networks. Submitted to Coastal Engineering 


Software and data for training deep convolutional neural network models to estimate wave height and wave period from surf zone imagery

This software was tested on Windows 10 with python 3.6, tensorflow 1.11.0 and keras 2.2.4. This software was written by Dr Daniel Buscombe at Northern Arizona University, in the winter of 2018/19.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. IF YOU ENCOUNTER A PROBLEM/BUG OR HAVE A QUESTION OR SUGGESTION, PLEASE USE THE "ISSUES" TAB ON GITHUB. OTHERWISE, THIS SOFTWARE IS UNSUPPORTED.


### Folder structure

* \conda_env contains yml files for setting up a conda environment
* \conf contains the configuration file with user-definable settings
* \train contains files using for training models 
* \im128 is a file structure that will contain results from model training

## Setting up computing environments

### Install Anaconda pyton distribution

Install the latest version of Anaconda (https://www.anaconda.com/distribution/)

When installed, open an anaconda command prompt

### Get this github repository

Use git to clone the github directory

```
git clone --depth 1 git@github.com:dbuscombe-usgs/OpticalWaveGauging_DNN.git
```

navigate to the ```OpticalWaveGauging_DNN``` directory

```
cd OpticalWaveGauging_DNN
```

It is strongly recommended that you use a GPU-enabled tensorflow installation. CPU training of a model can take several hours to several days (more likely the latter). However, the following instructions are for a CPU install. To use gpu, replace ```tensorflow``` with ```tensorflow-gpu``` in ```conda_env/owg.yml```


First, if you are a regular conda user, I would recommend some conda housekeeping (this might take a while):

```
conda clean --packages
conda update -n base conda
```

Otherwise (i.e. this is a fresh conda install), no housekeeping required.


### Create a conda virtual environment

Create a new conda environment called ```owg```

```
conda env create -f conda_env/owg.yml
```

This takes a few minutes. When it is done, activate environment:

```
conda activate owg
```

### Keras source code modifications

1. go to the keras_applications folder. On Windows anaconda builds this is typically located here:

```C:\Users\yourusername\AppData\Local\Continuum\anaconda3\envs\owg\Lib\site-packages\keras_applications```


2. Zip up all the .py files. Call the zipped file 'orig.zip'. Delete the .py files.

3. Then copy the .py files in conda_env\keras_applications to this directory


Finally, (if it is not already there) add the following code to the ```_init_.py``` file in the keras\applications folder, which is typically located here:

```C:\Users\yourusername\AppData\Local\Continuum\anaconda3\envs\owg\Lib\site-packages\keras\applications```


```
def keras_modules_injection(base_fun):

    def wrapper(*args, **kwargs):
        if hasattr(keras_applications, 'get_submodules_from_kwargs'):
            kwargs['backend'] = backend
            kwargs['layers'] = layers
            kwargs['models'] = models
            kwargs['utils'] = utils
        return base_fun(*args, **kwargs)

    return wrapper
```	



On Windows anaconda builds this is typically located here:

```C:\Users\yourusername\AppData\Local\Continuum\anaconda3\envs\owg\Lib\site-packages\keras\applications```


## Setting up the model 

Configuration files are in JSON format, like this:

```
{
  "samplewise_std_normalization" : true,
  "samplewise_center"  : true,
  "input_image_format" : "png"
  "input_csv_file"     : "IR-training-dataset.csv"
  "category"           : 'H',

  "horizontal_flip"    : false,
  "vertical_flip"      : false,
  "rotation_range"     : 10,
  "width_shift_range"  : 0.1,
  "height_shift_range" : 0.1,
  "shear_range"        : 0.05,
  "zoom_range"         : 0.2,
  "fill_mode"          : "reflect",
  
  "batch_size"         : 64,
  "img_size"           : 128,
  "num_epochs"         : 100,
  "test_size"          : 0.33,
  "steps_per_epoch"    : 100,
  "dropout_rate"       : 0.5,
  "epsilon"            : 0.0001,
  "min_lr"             : 0.0001,
  "factor"             : 0.8
}
```

### Training inputs

* imsize : size of image to use (pixels)
* category: 'H' for wave height, 'T' for wave period
* input_image_format: image file extension
* input_csv_file: name of file that has wave height and period per image

### Model hyperparameters

* num_epochs = number of training epochs
* test_size = proportion of data set to use for training
* batch_size = number of images to use per model training step
* steps_per_epoch = number of training steps per training epoch
* dropout_rate: proportion of neurons to randomly drop in dropout layer
* factor: factor by which the learning rate will be reduced. new_lr = lr * factor
* epsilon: threshold for measuring the new optimum, to only focus on significant changes.
* min_lr: lower bound on the learning rate.

### Image pre-processing

* samplewise_std_normalization: Bool. if True, Divide each input by its std.
* samplewise_center: Bool. if True, set each sample mean to 0.

### Image augmentation parameters:

* rotation_range: Int. Degree range for random rotations.
* width_shift_range: Float, 1-D array-like or int 
float: fraction of total width, if < 1, or pixels if >= 1.
1-D array-like: random elements from the array.
int: integer number of pixels from interval  (-width_shift_range, +width_shift_range)
With width_shift_range=2 possible values are integers [-1, 0, +1], same as with width_shift_range=[-1, 0, +1], while with width_shift_range=1.0 possible values are floats in the half-open interval [-1.0, +1.0[.
* height_shift_range: Float, 1-D array-like or int
float: fraction of total height, if < 1, or pixels if >= 1.
1-D array-like: random elements from the array.
int: integer number of pixels from interval  (-height_shift_range, +height_shift_range)
With height_shift_range=2 possible values are integers [-1, 0, +1], same as with height_shift_range=[-1, 0, +1], while with height_shift_range=1.0 possible values are floats in the half-open interval [-1.0, +1.0[.
* brightness_range: Tuple or list of two floats. Range for picking a brightness shift value from.
* shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
* zoom_range: Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
* fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. 
* horizontal_flip: Boolean. Randomly flip inputs horizontally.
* vertical_flip: Boolean. Randomly flip inputs vertically.

## Training models

To train models to predict wave height, the following scripts will do so for all combinations of 4 models (MobileNetV1, MobileNetV2, InceptionV3, and InceptionResnet2), and 4 batch sizes (16, 32, 64, and 128 images). 

```
python train_OWG.py
```

The following script does the same using generator functions in the training

```
python train_OWG_gen.py
```

Both scripts provide comparable results and are provided to illustrate two different options for training, mostly for advanced users wishing to modify and adapt the code for other purposes. 

You may notice ```python train_OWG_nogen.py``` is marginally faster

The best models are obtained using larger numbers of epochs (say, 100+), but you'll probably want to train them on a GPU (install ```tensorflow-gpu``` instead of ```tensorflow```).

To train OWGs for wave period, change the category in the config file to 'T' and run the above again

## Tidying up

Model result files (*.hdf5 format) are organized in the following file structure

im128

---res

------100epoch

---------H

------------model1

---------------batch16

---------------batch32

---------------batch64

---------------batch128

------------model2

---------------batch16

---------------batch32

---------------batch64

---------------batch128

------------model3

---------------batch16

---------------batch32

---------------batch64

---------------batch128

------------model4

---------------batch16

---------------batch32

---------------batch64

---------------batch128

---------T

------------model1

---------------batch16

---------------batch32

---------------batch64

---------------batch128

------------model2

---------------batch16

---------------batch32

---------------batch64

---------------batch128

------------model3

---------------batch16

---------------batch32

---------------batch64

---------------batch128

------------model4

---------------batch16

---------------batch32

---------------batch64

---------------batch128

Then run a script to split large model files to smaller files < 100 MB (so they fit on github)

```
python split_model4.py
```

Finally, compile and plot results from all models using

```
python compile_results.py
```

Data are written out to the Matlab format. For example, for the IR imagery wave height model, the mat file would be:

```
IR_all_model_preds_height_128.mat
```

and for the IR imagery wave period model, the mat file would be:

```
IR_all_model_preds_period_128.mat
```

## Operational Mode

### Testing model on a folder of images

```
python test_OWG_folder.py
```

This program will read the configuration file, conf/config_test.json

This file should be set up with the following information:

```
{
  "samplewise_std_normalization" : true,
  "samplewise_center"  : true,
  "weights_path" : "im128/res/100epoch/H/model1/batch16/waveheight_weights_model1_16batch.best.nearshore.hdf5",
  "input_csv_file"     : "train/snap-training-dataset.csv", 
  "category"           : "H",
  "im_size"            : 128,
  "image_direc"        : "train/snap_images"
}

```

where the ```image_direc``` is the folder where the test set of images are; ```weights_path``` is the hdf5 file associated with the model you wish to use; and ```input_csv_file``` should be a comma delimited file like the one used to train with.

The other variables, ```im_size```, ```category```, ```samplewise_std_normalization``` and ```samplewise_center``` are the same as used in model training


### Testing model on a single image

```
python predict_image.py -i path/to/image/file.ext
```

for example:

```
python predict_image.py -i train/snap_images/1516401000.cx.snap.jpg
```

The following variables are also read from the conf/config_test.json file: ```weights_path```, ```im_size```, ```category```, ```samplewise_std_normalization``` and ```samplewise_center```. These should be the same as used in model training


Deactivate environment when finished:

```
conda deactivate
```

