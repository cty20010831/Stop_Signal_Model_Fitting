# Stop_Signal_Model_Fitting

This repository serves as the testing base for implementing the [**stop signal respond task**](https://cambridgecognition.com/stop-signal-task-sst/) in the [`pymc`](https://www.pymc.io/welcome.html) ecosystem, before incorporating into the [`hssm`](https://lnccbrown.github.io/HSSM/) architecture (also built upon `pymc`).

This project is in collaboration with Michael J. Frank and Alex Fengler from Brown University.

## Requirements
It is recommended to run the code on the local machine by setting up a virtual environment (e.g., [venv](https://docs.python.org/3/library/venv.html) or [conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)). For example, you can create the virtual `venv`
environment named **venv**:
```bash
# Create the virtual environment:
python3.11 -m venv venv

# Activate the virtual environment:
source venv/bin/activate
```

Then, download the required packages:
```bash
pip install -r requirements.txt
```

The [requirements.txt](requirements.txt) is generated using [pipreqs](https://github.com/bndr/pipreqs) package, which is a convenient package to generate requirements.txt file based on imports:
```bash
pipreqs --scan-notebooks
```

## (Top-Level) Repository Structure
    .
    ├── model_fitting/        # hierarchical and individual level model fitting
    ├── simulation/           # (forward) simulator of stop signal test
    ├── .gitignore
    ├── README.md
    └── requirementd.txt   

### `model_fitting` directory
1. [model_fitting_simple_test_hierarchical_no_p_tf.ipynb](model_fitting/model_fitting_simple_test_hierarchical_no_p_tf.ipynb): fit a small sample of participants (with varied trials per participant) without *p_tf* at the hierarchical level
2. [model_fitting_simple_test_hierarchical.ipynb](model_fitting/model_fitting_simple_test_hierarchical.ipynb): fit a small sample of participants (with varied trials per participant) at the hierarchical level
3. [model_fitting_simple_test_individual.ipynb](model_fitting/model_fitting_simple_test_individual.ipynb): fit a small sample of participants (with varied trials per participant) at the individual level (both with and without *p_tf*) and test how varying stop parameters and number of trials affect posterior distributions. 
4. [random.py](model_fitting/random.ipynb): some debugging tries on examining the influences of varying parameters (used in formward simulation) on the posterior distribution of parameters.
5. [test_custom_likelihood_archived.ipynb](model_fitting/test_custom_likelihood_archived.ipynb): archived notebook test whether likelihood defintion in [util_archived.py](model_fitting/util_archived.py) is correct. 
6. [util_archived.py](model_fitting/util_archived.py): archived utility functions used in model fitting notebooks (likelihood defined using `PyTensor Op`). 
7. [util.py](model_fitting/util.py): utility functions used in model fitting notebooks, including likelihood defintion for different trial types and posterior predictive sampling (check). 

### `simulation` directory
1. [sanity_check.ipynb](simulation/sanity_check.ipynb): sanity check of (forward) simulator
2. [simulate_hierarchical_pymc.py](simulation/simulate_hierarchical_pymc.py): simulate data for hierarchical-level model fitting (with group/hyper parameters)
3. [simulate_individual_pymc.py](simulation/simulate_individual_pymc.py): simulate data for individual-level model fitting (without group/hyper parameters)
4. [util.py](simulation/util.py): utility functions for forward simulator, including generating a random value following Ex-Gaussian distribution and simulating one synthetic experiment round of trials for a subject (for fixed and staircase ssd).

## Google Drive
A Google Drive folder has been created to store saved traces and saved trials for posterior predictive check. To download them to the local directory: 
```bash
cd model_fitting

# Download saved traces
gdown --no-check-certificate --folder https://drive.google.com/drive/u/0/folders/1Mgy8nQKrI3nMAhqskP0pfwt6dCyDWDoJ?ths=true

# Download saved trials
gdown --no-check-certificate --folder https://drive.google.com/drive/u/0/folders/1iHmZUOqJilN5Xudgk6qY-NVhwykKvk3w?ths=true

cd ..
```