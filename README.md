# IR_wavegauge
Data and code to implement Buscombe and Carini (2019) optical wave gauging and classification using deep neural networks

## About

Software and data for training deep convolutional neural network models to:

1. Classify wave breaker type from IR images of breaking waves in the surf zone
2. Estimate wave height and wave period from the same imagery

### Folder structure

* \conda_env contains yml files for setting up conda environments (one each for discrete classification and continuous regression)
* \conf contains configuration files using for training models for discrete classification
* \train contains files using for training models for discrete classification and continuous regression
* \test contains files using for testing models for discrete classification and continuous regression
* \out contains output files from model training
* \keras_mods contains modified keras applications files for discrete classification and continuous regression

## Setting up computing environments

These two tasks require different modifications to keras libraries. The easiest way to deal with two different keras installs is to use conda environments

First, some conda housekeeping

```
conda clean --packages
conda update -n base conda
```

### Discrete classification: estimating wave breaker type from imagery

1. Create a new conda environment called ```classification```

```
conda env create -f conda_env/classification.yml
```
C:\Users\ddb265\github_clones\IR_wavegauge\

2. Copy the contents of the ```keras_mods\classification\tf_python_keras_applications``` folder into the ```tensorflow\python\keras\applications``` site package in your new conda env path. For example: 

```
C:\Users\user\AppData\Local\Continuum\anaconda3\envs\classification\Lib\site-packages\tensorflow\python\keras\applications
```

Be sure to keep a copy of the existing files there in case something goes wrong.

3. Copy the contents of the ```keras_mods\classification\tf_keras_applications``` folder into the ```tensorflow\keras\applications``` site package in your new conda env path. For example: 

```
C:\Users\user\AppData\Local\Continuum\anaconda3\envs\classification\Lib\site-packages\tensorflow\keras\applications
```

4. Activate environment:

```
conda activate classification
```



Deactivate environment when finished:

```
conda deactivate
```


### Regression: estimating wave height/period from imagery

It is strongly recommended that you use a GPU-enabled tensorflow installation for the regression task. CPU training of a model can take several hours to several days. However, the following instructions are for a CPU install. To use gpu, replace ```tensorflow``` with ```tensorflow-gpu``` in ```conda_env/regression.yml```

Create a new conda environment called ```regression```

```
conda env create -f conda_env/regression.yml
```

From the ```keras_mods\regression\keras_applications``` folder, copy ```inception_v3.py``` and ```inceptionresnet_v2.py``` into your ```keras_applications``` folder within your conda environment. For example: 

```C:\Users\user\AppData\Local\Continuum\anaconda3\envs\regression\Lib\site-packages\keras_applications```

Be sure to keep a copy of the existing files there in case something goes wrong.

Activate environment:

```
conda activate regression
```

Install ```pillow``` using ```pip``` (because the conda version was incompatible with conda-installed ```tensorflow```, at least at time of writing)

```
pip install pillow
```

Deactivate environment when finished:

```
conda deactivate
```



## Training classification models

### Extract image features 

The following has been tested with the following models: MobileNetV1, MobileNetV2, Xception, InceptionV3, InceptionResnet2, and VGG19

1. Run the feature extractor using the MobileNetV2 model, with augmented images, running ```extract_features_imaug.py``` and the configuration file ```conf/conf_mobilenet.json```:

```
python extract_features_imaug.py -c conf_xception
```

2. Run the feature extractor using the Xception model, without augmented images, running ```extract_features.py``` and the configuration file ```conf/conf_xception.json```:

```
python extract_features.py -c conf_mobilenet
```

### Train and save model


### Test model



## Training regression models

To train models to predict wave height, the following script will do so for all combinations of 4 models (MobileNetV1, MobileNetV2, InceptionV3, and InceptionResnet2), and 4 batch sizes (16, 32, 64, and 128 images). Model training is limited to 20 epochs. Better models are obtained using 100 epochs, but you'll probably want to train them on a GPU (install ```tensorflow-gpu``` instead of ```tensorflow```).

```
python train_regress_height.py
```

To do the same for wave period:

```
python train_regress_period.py
```

