#!/bin/bash

pip install virtualenv
virtualenv codo-env
source codo-env/bin/activate
pip install -r requirements.txt
python setup.py clean
python -m ipykernel install --user --name=codo-env