sample project with conda environment
==============================

Create new environment: conda env create -f environment.yml -n ml-env
Update environment: conda env update -f environment.yml -n ml-env


## Overview

This project uses:

* `conda` for managing the development environment.
* `kedro` for structuring the source code and organising data pipelines.


## A Note About Git Tracking

The `.gitignore` file of this repository has been configured to avoid tracking the following files:

* Any files whose names contain the word `credential`.
* All files in the `conf/local` folder.
* All files in the `data` folder (you can use this folder to stored temporary files for your experiments).


### 1. Managing the Development Environment

To initialise the development environment, run the following `conda` command at the project's root directory:

```
conda env create --prefix ./.conda-env --file environment.yml
conda activate ./.conda-env
```

To update the environment after changing the 'environment.yml' file, use the following command:

```
conda env update --prefix ./.conda-env --file environment.yml --prune
```

Or simply, rebuild the environment from scratch:

```
conda env create --prefix ./.conda-env --file environment.yml --force
```


### 2. Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


### 3. Streamlit
pip install streamlit
streamlit hello

Fixing error in Streamlit:
- https://discuss.streamlit.io/t/toml-docoder-error/1400/39
- In windows, go to the C:/Windows/Users/%profile name%/.streamlit, delete the toml files and restart streamlit run app.py
- For those who fail the deployment on Heroku. You might found several tutorials suggesting the setup.sh, but please have a look when you copy-paste the code. I deploy the website without deleting the config.toml file. 

1. Create file multiplage.py
2. Create folder page
3. Create analysis app files in page folder
4. Import analysis app files in file app.py



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
