## About

Data and code to implement optical wave gauging (OWG) using deep neural networks, detailed in the paper Buscombe et al (2019):

> Buscombe, Carini, Harrison, Chickadel, and Warrick (2019) Optical wave gauging using deep neural networks. Coastal Engineering  https://doi.org/10.1016/j.coastaleng.2019.103593

Software and data for training deep convolutional neural network models to estimate wave height and wave period from surf zone imagery

This software was tested on Windows 10 and Ubuntu Linux with python 3.7, tensorflow 2. This software was written by Dr Daniel Buscombe at Northern Arizona University, 2018-2019.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. IF YOU ENCOUNTER A PROBLEM/BUG OR HAVE A QUESTION OR SUGGESTION, PLEASE USE THE "ISSUES" TAB ON GITHUB. OTHERWISE, THIS SOFTWARE IS UNSUPPORTED.


### Folder structure

* \conda_env contains yml files for setting up a conda environment
* \config contains the configuration file with user-definable settings
* \train contains files using for training models 
* \im128 is a file structure that will contain results from model training

## Setting up computing environments

### Install Anaconda python distribution

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

(If you get an error related to installing ```tensorflow```, replace ```tensorflow-gpu==2.0``` with ```tensorflow-gpu``` or simply the CPU version ```tensorflow``` )

This takes a few minutes. When it is done, activate environment:

```
conda activate owg
```


## A note on versions and releases

This is an evolving project and things move fast in the world of deep learning. During the 8 months the Coastal Engineering paper associated with this repository was in review, Tensorflow and Keras libraries underwent some major changes. Tensorflow did a major upgrade from version 1 to version 2, and keras got subsumed as into Tensorflow as tf.keras.

![This](https://github.com/dbuscombe-usgs/OpticalWaveGauging_DNN/releases/tag/11.11.19) is the version of the software that worked in October 2019 prior to the official release of Tensorflow 2, and may be used to exactly reproduce the results of the paper, if you can get it to work using tensorflow 1.X. The  release version "11.11.19" will not work in tensorflow 2.X.

Later releases, including this release, work in Tensorflow 2 using ```tf.keras```


## Setting up the model 

Configuration files are in JSON format, like this:

```
{
  "samplewise_std_normalization" : true,
  "samplewise_center"  : true,
  "input_image_format" : "jpg",
  "input_csv_file"     : "snap-training-dataset.csv", 
  "category"           : "H",
  "prc_lower_withheld": 5,
  "prc_upper_withheld": 5,
  
  "horizontal_flip"    : false,
  "vertical_flip"      : false,
  "rotation_range"     : 10,
  "width_shift_range"  : 0.1,
  "height_shift_range" : 0.1,
  "shear_range"        : 0.05,
  "zoom_range"         : 0.2,
  "fill_mode"          : "reflect",
  
  "img_size"           : 128,
  "num_epochs"         : 5,
  "test_size"          : 0.4,
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

If you do not wish to retrain all the OWGs yourself from scratch, MobileNetV1-based OWGs (only) are provided. See ```im128/res/200epoch/H/model1``` for wave height and ```im128/res/200epoch/T/model1``` for wave period. The hdf5 format files contain the model's weights. The .json format files contain the model architecture. The image files show training outputs and are automatically generated by ```python train_OWG.py```, described below.

To train models  yourself (recommended), the following scripts will do so for all combinations of 4 models (MobileNetV1, DenseNet201, InceptionV3, and InceptionResnet2), and 4 batch sizes (16, 32, 64, and 128 images). 

Note that in this Tensorflow 2 version of the repository, DenseNet201 is used instead of MobileNetV2, which was used in the paper and the original repository (see release version 11.11.18) but is not implemented correctly at the time of writing using keras in Tensorflow 2 without modifying the MobileNetV2 source script. Therefore DenseNet201 is used instead, which gives similar results to MobileNetV2

```
python train_OWG.py -c configfile.json
```

In the above, ```configfile.json``` is one of the config files in the . Just provide the name of the json file, including the 'json' file extension, not the full path to the file, like this:

```
python train_OWG.py -c config_IR_H.json
```

The best models are obtained using larger numbers of epochs (say, 100+), but you'll probably want to train them on a GPU (install ```tensorflow-gpu``` instead of ```tensorflow```).

To train OWGs for wave period, change the category in the config file to 'T' and run the above again

## Tidying up

Model result files (*.hdf5 format, which contain the model's weights) are organized in the following file structure

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


Finally, compile and plot results from all models using

```
python compile_results.py
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


## Wrapping up

Deactivate environment when finished:

```
conda deactivate
```

