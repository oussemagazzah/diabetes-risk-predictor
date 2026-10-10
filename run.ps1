# Windows equivalent of the Makefile:  .\run.ps1 train | evaluate | all | test | lint | app | eda
param([Parameter(Mandatory = $true)][string]$Task)
$ErrorActionPreference = "Stop"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

switch ($Task) {
    "install"  { & $py -m pip install -r requirements.txt; & $py -m pip install -e . }
    "train"    { & $py -m diabetes_predictor.train }
    "evaluate" { & $py -m diabetes_predictor.evaluate; & $py -m diabetes_predictor.explain }
    "all"      { & $py -m diabetes_predictor.train; & $py -m diabetes_predictor.evaluate; & $py -m diabetes_predictor.explain }
    "test"     { & $py -m pytest -q }
    "lint"     { & $py -m ruff check .; & $py -m ruff format --check . }
    "app"      { & $py -m streamlit run app/streamlit_app.py }
    "eda"      { & $py -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb }
    default    { Write-Error "Unknown task: $Task" }
}
