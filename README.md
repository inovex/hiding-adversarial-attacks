# Hiding Adversarial Attacks on Convolutional Neural Networks Using Manipulated Explanations

This is the repository to my M.Sc. thesis where I combined adversarial attacks on Fashion-MNIST images with an adversarial
fine-tuning procedure to manipulate the visual explanations produced by Grad-CAM and Guided Backpropagation.

## Description

The three goals of my thesis were to...
1) create visual explanations for the adversarial images after the fine-tuning that look similar to the explanations of the original
images before the fine-tuning,
2) keep the network's classification performance on the original images approximately stable, and
3) ensure that the adversarial images are consistently misclassified.

## Setup
### Conda environment

In order to set up the conda environment:

1. Clone the repo and cd into the project directory, then create the `hiding_adversarial_attacks` environment with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. Activate the new environment with:
   ```
   conda activate hiding_adversarial_attacks
   ```

### Download the model weights
1. Download the model weights from [Google Drive]() => add url
2. Extract and move them to the "models" directory.

The models made available here are a subset of the ones used in my thesis. Please open an issue should you require more.


### (Optional) Logging to Neptune
This project enables you to use [Neptune.ai](https://neptune.ai) for logging training and test results to the cloud.
However, this is turned off by default so that the logs will be saved locally. If you're fine with this, you do not need to change anything.

If you want to use Neptune for logging, head over to their [website](https://neptune.ai), create an account and a project.
Note down the project name and API key and export them as environment variables:

```
   export NEPTUNE_API_TOKEN=<your-token>
   export NEPTUNE_PROJECT=<your-username/your-project-name>
```


## Running the code
The whole process consists of the following 4 steps, all of which are configurable:
1) Training the Fashion-MNIST classifier. You can skip this and just use the pre-trained weights.
2) Downloading and adversarially attacking the Fashion-MNIST data set. You can also skip this and download the attacked data.
3) Creating the initial visual explanation maps. Can also be skipped.
4) Running the adversarial fine-tuning based on the model from step 1.

### 1) Training the Fashion-MNIST classifier (optional)
> If you want to skip this, please download the model weights as described previously.



## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── hiding_adversarial_attacks <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.0 and the [dsproject extension] 0.6.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
