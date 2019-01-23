# IR_wavegauge
Data and code to implement Buscombe and Carini (2019) optical wave gauging and classification using deep neural networks

### Folder structure

* \conda_env contains yml files for setting up conda environments (one each for discrete classification and continuous regression)
* \conf contains configuration files using for training models for discrete classification
* \train contains files using for training models for discrete classification and continuous regression
* \test contains files using for testing models for discrete classification and continuous regression
* \out contains output files from model training


### Regression: estimating wave height/period from imagery

```
conda env create -f conda_env/env.yml
```


### Discrete classification: estimating wave breaker type from imagery

```
conda env create -f conda_env/env.yml
```

