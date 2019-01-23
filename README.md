# IR_wavegauge
Data and code to implement Buscombe and Carini (2019) optical wave gauging and classification using deep neural networks

### Folder structure

* \conda_env contains yml files for setting up conda environments (one each for discrete classification and continuous regression)
* \conf contains configuration files using for training models for discrete classification
* \train contains files using for training models for discrete classification and continuous regression
* \test contains files using for testing models for discrete classification and continuous regression
* \out contains output files from model training
* \keras_mods contains modified keras applications files for discrete classification and continuous regression

### Regression: estimating wave height/period from imagery

```
conda env create -f conda_env/env.yml
```

C:\Users\ddb265\AppData\Local\Continuum\anaconda3\envs\regression\Lib\site-packages\keras_applications

replace ```inception_v3.py``` and ```inceptionresnet_v2.py``` into your ```keras_applications``` folder with those in the 


### Discrete classification: estimating wave breaker type from imagery

```
conda env create -f conda_env/env.yml
```

