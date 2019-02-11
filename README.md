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

## Tidying up

Each model

Organize model result files (*.hdf5 format) in the following file structure

im128
---res_snap
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

(run twice using ```cat = 'H'``` and ```cat = 'T'```)


Data are written out to the Matlab format:

IR_all_model_preds_height_128.mat

and

IR_all_model_preds_period_128.mat


