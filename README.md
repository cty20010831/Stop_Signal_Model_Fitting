# Stop_Signal_Model_Fitting

This repository serves as the testing base for implementing the [**stop signal respond task**](https://cambridgecognition.com/stop-signal-task-sst/) in the [`pymc`](https://www.pymc.io/welcome.html) ecosystem, before incorporating into the [`hssm`](https://lnccbrown.github.io/HSSM/) architecture (also built upon `pymc`).

This project is in collaboration with Michael J. Frank and Alex Fengler from Brown University.

## Requirements
It is recommended to run the code on the local machine by setting up a virtual environment (e.g., [venv](https://docs.python.org/3/library/venv.html) or [conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)) that downloads the required packages:
```bash
pip install -r requirements.txt
```

The [requirements.txt](requirements.txt) is generated using [pipreqs](https://github.com/bndr/pipreqs) package, which is a convenient package to generate requirements.txt file based on imports:
```bash
pipreqs --scan-notebooks
```