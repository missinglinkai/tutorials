# MissingLinkAI SDK examples for scikit-learn

## Requirements

You need Python 2.7 or 3.5 on your system to run this example.

To install the dependency:
- You are strongly recommended to use [`virtualenv`](https://virtualenv.pypa.io/en/stable/) to create a sandboxed environment for individual Python projects
```bash
pip install virtualenv
```

- Create and activate the virtual environment
```bash
virtualenv .venv
source .venv/bin/activate
```

- Install dependency libraries
```bash
pip install -r requirements.txt
```

## Run

In order to run an experiment with MissingLinkAI, you would need to first create a
project

```bash
# In case you havn't done this before connect your MissingLink account 
ml auth init

python mnist.py
```