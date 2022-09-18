# PPO Sandbox & Experiments

## Getting started

PPO sandbox uses Poetry for python package management.

Installation instructions for Poetry can be found at https://python-poetry.org/docs/

Install packages and run:

```shell
poetry install
poetry run python ppo_from_scratch.py
```

Many parameters are configurable from the command line. For an overview, see:

```shell
poetry run python ppo_from_scratch.py --help
```

## Troubleshooting

In some cases you may encounter issues installing `gym[box2d]`, related to swig.
First, check that swig is installed:

```shell
# MacOS
brew install swig

# Ubuntu / Debian
sudo apt install swig
```

Then manually install box2d with pip in the virtual env:

```shell
poetry run pip install gym[box2d]
```
