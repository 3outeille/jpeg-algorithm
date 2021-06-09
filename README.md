# Setup

- Create a virtual environment in the root folder using [virtualenv][virtualenv] and activate it.

```bash
# On Linux terminal, using virtualenv.
virtualenv codo-env
# Activate it.
source codo-env/bin/activate
pip install -r requirements.txt
```

- Add `codo-env` to jupyter notebook:
```
python -m ipykernel install --user --name=codo-env
```
- Remove `codo-env` from jupyter notebook:
```
jupyter kernelspec uninstall codo-env
```