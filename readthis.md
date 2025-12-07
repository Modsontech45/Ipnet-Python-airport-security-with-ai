## when you add a new libs make sure you add it to the libs in requiremnent
## 1 first create virtuall environment
```bash
python -m venv venv
```
## 2 activate it 
## Windows (PowerShell)
```bash
.\venv\Scripts\Activate.ps1
```
## desactivate it
``` bash
deactivate
```

## Method 1: Using a Virtual Environment

1. Make sure you are inside your virtual environment.
2. Run the following command in your terminal:

```bash
pip freeze > requirements.txt

```

## Sometimes pip freeze includes unnecessary packages. To generate a cleaner list
1. Install pipreqs
```bash
pip install pipreqs
```
1. Run pipreqs in your project folder

``` bash
pipreqs .
``` 

## now Installing Dependencies from requirements.txt
``` bash 
pip install -r requirements.txt
```