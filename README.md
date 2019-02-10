### About

Data and code to implement Buscombe et al (2019) optical wave gauging using deep neural networks, detailed in the paper

> Buscombe, Carini, Harrison, Chickadel, and Warrick (in review) Deep optical wave gauging. Submitted to Coastal Engineering 


Software and data for training deep convolutional neural network models to estimate wave height and wave period from the same imagery

### Folder structure

* \conda_env contains yml files for setting up conda environments (one each for discrete classification and continuous regression)
* \train contains files using for training models 

### Setting up computing environments

First, some conda housekeeping

```
conda clean --packages
conda update -n base conda
```

It is strongly recommended that you use a GPU-enabled tensorflow installation for the regression task. CPU training of a model can take several hours to several days (more likely the latter). However, the following instructions are for a CPU install. To use gpu, replace ```tensorflow``` with ```tensorflow-gpu``` in ```conda_env/regression.yml```

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

## Training models

To train models to predict wave height, the following script will do so for all combinations of 4 models (MobileNetV1, MobileNetV2, InceptionV3, and InceptionResnet2), and 4 batch sizes (16, 32, 64, and 128 images). Better models are obtained using 100 epochs, but you'll probably want to train them on a GPU (install ```tensorflow-gpu``` instead of ```tensorflow```).

```
python train_regress_height.py
```

To do the same for wave period:

```
python train_regress_period.py
```

