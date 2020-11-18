echo "Updating the PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Installing dependancies"
poetry install --remove-untracked

echo "Activating vitrual environment"
poetry shell
