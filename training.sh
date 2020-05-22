#!/bin/bash


function pre_flight_checks(){
if [[ "$VIRTUAL_ENV" = "" ]]; then
  echo "not in venv"
  exit 1
fi

echo "running sanity checks ..."
echo "running mypy ..."
python -m mypy .

echo "running tests ..."
PYTHONPATH=neuro_evolution_ctrnn pytest -q --disable-pytest-warnings tests

echo "checking git-repo ..."
git fetch
if [ -n "`git rev-list HEAD..origin`"] ; then
  echo "Error: Upstream changes in git repo. Please do 'git pull --rebase' before running simulations"
  exit 1
fi

if [ -n "$(git status --porcelain |grep -v '.json')"] ; then
  echo "Error: Pending changes in Git Repo, that are not config-files. Please commit changes before running simulations"
  exit 1
fi

if [ -n "`git rev-list origin..HEAD`"] ; then
  echo "Error: Pleaes push local changes before running simulations"
  exit 1
fi

echo "all checks passed."
}

source $HOME/.venv/neuro/bin/activate

pre_flight_checks

echo "starting simulation ..."
python -m scoop neuro_evolution_ctrnn/train.py "$@"
