# PPO Sandbox & Experiments

## Getting started

PPO sandbox uses poetry for python package management. Installation instructions can be found at https://python-poetry.org/docs/

Then install packages and run:

```
poetry install
poetry run python ppo_from_scratch.py
```

Many parameters are configurable from the command line. For an overview, see:

```
poetry run python ppo_from_scratch.py --help
```

## Troubleshooting

In some cases you may encounter issues installing `gym[box2d]`, related to swig.
First, check that swig is installed:

```
# MacOS
brew install swig

# Ubuntu / Debian
sudo apt install swig
```

Then manually install box2d in the venv:

```
poetry run pip install gym[box2d]
```
