# Hiding Adversarial Attacks on Convolutional Neural Networks Using Manipulated Explanations

This is the repository to my M.Sc. thesis where I combined adversarial attacks on Fashion-MNIST images with an adversarial
fine-tuning procedure to manipulate the visual explanations produced by the explainability techniques [Grad-CAM](https://arxiv.org/abs/1610.02391)
and [Guided Backpropagation](https://arxiv.org/abs/1412.6806).

## Description
eXplainable Artificial Intelligence (XAI) has seen a hype in the last few years due to the black-box nature
 of many deep learning models. We want to better understand the decision-making process of our models and which input features influence the output.
Also, some have suggested that explainability techniques could be used for auditing machine learning models, e.g. to check whether
they exhibit biases.
However, researchers have recently found out that some explainability techniques can be manipulated to produce explanations for images that have nothing to do
with the image content (see the papers by [Dombrowski et al.](https://arxiv.org/abs/1906.07983), [Ghorbani et al.](https://arxiv.org/abs/1710.10547)
and [Heo et al.](https://arxiv.org/abs/1902.02041)).

In my thesis I empirically investigated whether it is possible to hide a manipulation of the input data from a
potential audit that uses explainability techniques. The idea behind this is that a malicious actor could try to make
a visual explanation of a manipulated image look very similar to the explanation of the non-manipulated counterpart.
This way, an auditor would not be able to tell that a manipulation has taken place by examining the explanations.

To achieve this, I first used an established adversarial attack technique called [DeepFool](https://arxiv.org/abs/1511.04599)
to manipulate the original images in the Fashion-MNIST data set in order to create adversarial images that are
misclassified by a pre-trained CNN model. Then, I came up with a training procedure, also called adversarial fine-tuning, that is based
on the original and adversarial images, as well as their explanation maps that are created using the two techniques Grad-CAM and
Guided Backpropagation.
The three goals of the fine-tuning were to...
1) create visual explanations for the adversarial images after the fine-tuning that look similar to the explanations of the original
images before the fine-tuning,
2) keep the network's classification performance on the original images approximately stable, and
3) ensure that the adversarial images are consistently misclassified after the fine-tuning.

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
1. Download the whole directory called `weights` from [Google Drive](https://drive.google.com/drive/folders/1KaxOrPmgUJgIWKTyPXsYhwyKqZL8kBtK?usp=sharing).
2. Extract the zip file inside the project's `models` directory.

You should now see two directories: `pre-manipulation` and `post-manipulation`. The first one contains the pre-trained Fashion-MNIST classifier
checkpoint `fashion-mnist-model.ckpt` that everything builds upon. If you just want to use these, you can skip step 1. of `Running the code`.
The weights in the `post-manipulation` directory are the results of my thesis. They can be used in step 4.

The models made available here are a subset of the ones used in my thesis. Please open an issue should you require more.

### Download the Fashion-MNIST data set
1. Download the Fashion-MNIST data set files `training.pt` and `test.pt` from [Google Drive](https://drive.google.com/drive/folders/1ctLZOums9xa-brc6DDVCoO9dpXvH3d0e?usp=sharing).
2. Move the files inside the project directory `data/external/FashionMNIST/processed`. The final result should look something like this:
```
├── data
│   ├── external
|   |   └──FashionMNIST
|   |   |   └──processed
|   |   |   |   ├──test.pt
│   │   |   |   └──training.pt
```

### (Optional) Logging to Neptune
This project enables you to use [Neptune.ai](https://neptune.ai) for logging training and test results to the cloud.
However, this is turned off by default so that the logs will be saved locally. If you're fine with this, you do not need to change anything.

If you want to use Neptune for logging, head over to their [website](https://neptune.ai), create an account and a project.
Note down the project name and API key and export them as environment variables:

```
   export NEPTUNE_API_TOKEN=<your-token>
   export NEPTUNE_PROJECT=<your-username/your-project-name>
```
When running step 1 and step 4 below, add the following to the Python commands: `neptune_offline_mode=False`.

Logs are also saved locally, together with other training and testing output such as images and csv files.
All logs are saved to the project's `logs` directory, but each of the following steps creates their own subdirectory.


## Running the code
The whole process consists of the following 4 steps, all of which are configurable:
1) Training the Fashion-MNIST classifier (optional: you can skip this and just use the pre-trained weights).
2) Adversarially attacking the Fashion-MNIST data set (optional: you can also skip this and download the attacked data).
3) Creating the initial visual explanation maps (optional: can also be skipped and the data downloaded).
4) Running the adversarial fine-tuning based on the pre-trained model, the adversarial images and the visual explanations.

### 1) Training the Fashion-MNIST classifier
> If you want to skip this, please download the model weights in the `pre-manipulation` directory as described previously.

#### Quick
If you want to replicate the exact results, you first need to download the Fashion-MNIST data set as described above.

#### Not so quick
Activate the conda environment, cd into the `src/hiding_adversarial_attacks` directory and
run the following:
```
python train_classifier.py data_set=FashionMNIST classifier=FashionMNISTClassifier
```
This will train a Fashion-MNIST CNN classifier for 25 epochs using a learning rate of 1.5 with the Adadelta optimizer, static learning rate decay factor of 0.85 and batch size of 64.
You can check out other configuration presets for the training in the file `src/hiding_adversarial_attacks/config/classifier_training_config.py`.

Logs and model checkpoints are saved to the directory `logs/train_classifier`.

To test your model run the following command and add the path to your model checkpoint:
```
python train_classifier.py \
test=True \
data_set=FashionMNIST \
classifier=FashionMNISTClassifier \
checkpoint=<path-to-checkpoint>
```
> ⚠️ Beware that all special characters (e.g. "=") need to be escaped with a forward slash ("\")!

### 2) Adversarially attacking the Fashion-MNIST data set
#### Quick
If you want to skip this step, download all of the 8 PyTorch tensors (*.pt) from [Google Drive](https://drive.google.com/drive/folders/1tVO8N7LQPYQ12tElFMsexjCn5UiBrxjk?usp=sharing).
Move the *.pt files inside the local project directory `data/preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps=0.105--cp-run=HAA-1728`.

#### Not so quick
If you want to create your own adversarial data, cd into the `src/hiding_adversarial_attacks` directory.
Run the following command, adding the path to your trained FashionMNIST checkpoint (*.ckpt) file and
a custom identifier that will be appended to the directory where the resulting tensors will be stored after the attack
(I used the ID of the Neptune run of the model training (HAA-1728), but you're free to choose whatever you like):
```
python run_attack_on_data.py \
data_set=FashionMNIST \
classifier=FashionMNISTClassifier \
checkpoint=<path-to-checkpoint> \
checkpoint_run=<custom-checkpoint-identifier> \
attack.epsilons=[0.105]
```
This will run an adversarial attack called DeepFool on the Fashion-MNIST data set and create 8 PyTorch tensors in a directory at `data/preprocessed/adversarial`.

You can also change the `attack.epsilons` parameter by replacing or adding different values.
This will affect the adversarial attack size: the larger the values, the more visible the changes will be in the attacked images.
For more configuration options, see the file `src/hiding_adversarial_attacks/config/adversarial_attack_config.py`.

Logs and model checkpoints are saved to the directory `logs/run_attack_on_data`.

### 3) Creating the initial visual explanation maps
#### Quick
If you want to skip this step, you can download the pre-created explanation maps from Google Drive. Please download both directories
`exp=GradCAM--l=conv2--ra=False` and `exp=GuidedBackprop` as *.zip files from [here](https://drive.google.com/drive/folders/19nIg1eOwpT5nkLKihW5iUSDKITB4r8HO?usp=sharing) and unzip them inside `data/preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps=0.105--cp-run=HAA-1728`.

#### Not so quick
If you want to create the explanation maps yourself, first cd into the `src/hiding_adversarial_attacks` directory.
Run the following command for creating the Grad-CAM explanation maps, adding the path to your Fashion-MNIST classifier checkpoint and the directory containing the results
of the adversarial attack from step 2 (in case you downloaded the data, set this to `data_path=<path-to-local-project-root>/data/preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps=0.105--cp-run=HAA-1728`).
```
python create_initial_explanations.py \
explainer=GradCamExplainer \
classifier=FashionMNISTClassifier \
data_set=AdversarialFashionMNIST \
checkpoint=<path-to-checkpoint> \
data_path=<path-to-adversarial-data-set-from-step-2>
```
For creating Guided Backpropagation explanations, replace `explainer="GradCamExplainer"` with `explainer="GuidedBackpropExplainer"`.
Also check out the file `src/hiding_adversarial_attacks/config/create_explanations_config.py` for more configuration options and additional explainability algorithms.

Logs and model checkpoints are saved to the directory `logs/explanation`.

### 4) Running the adversarial fine-tuning
> ⚠️ Please make sure you followed the steps before (at least downloading the required data & model checkpoint).

#### Quick (testing only)
If you just want to run the test stage on the manipulated models provided (see `Download the model weights` above),
then you can run the following code to test the manipulation for Grad-CAM and Fashion-MNIST class Sandal (ID: 5):

```
python train_manipulated_model.py \
test=True explainer="GradCamExplainer" similarity_loss="PCC" \
data_set="AdversarialFashionMNISTWithExplanations" classifier="FashionMNISTClassifier" \
data_path=<path-to-the-explanations-directory-from-step-3> \
explanations_path=<path-to-the-explanations-directory-from-step-3> \
classifier_checkpoint=<path-to-checkpoint> \
checkpoint=["<path-to-local-project-root>/models/post-manipulation/Grad-CAM/Sandal/final-model.ckpt"] \
included_classes=[5] \
normalize_explanations=True \
gpus=0
```
Make sure you add the correct paths for `data_path`, `explanations_path`, `classifier_checkpoint` (checkpoint of original model from step 1) and `checkpoint` (path to manipulated model that you downloaded).
If you have a GPU available you can speed things up by setting `gpus=1`.

If you want to run the test for class Coat, make sure to change the checkpoint to `checkpoint=["<path-to-local-project-root>/models/post-manipulation/Grad-CAM/Coat/final-model.ckpt"]`
and set `included_classes=[4]`.

For testing Guided Backpropagation, change the explainer to `explainer="GuidedBackpropExplainer"` and specify the
corresponding model checkpoint in the `models/post-manipulation` directory.


#### Not so quick (training and testing)
To replicate the adversarial manipulation of Grad-CAM explanations of class Sandal that I used in my thesis, you can use the
following command:

```
python train_manipulated_model.py \
classifier="FashionMNISTClassifier" \
data_set="AdversarialFashionMNISTWithExplanations" \
explainer="GradCamExplainer" \
similarity_loss="PCC" \
data_path=<path-to-the-explanations-directory-from-step-3> \
explanations_path=<path-to-the-explanations-directory-from-step-3> \
classifier_checkpoint=<path-to-checkpoint> \
included_classes=[5] \
max_epochs=30 \
lr=0.00003 \
gamma=0.7 \
steps_lr=5 \
weight_decay=0.0 \
loss_weight_similarity=1.0 \
ce_class_weight=1 \
batch_size=64 \
normalize_explanations=True \
convert_to_softplus=True \
gpus=0
```

If you have a GPU available, you can change the last line to `gpus=1`.

The table below specifies the hyperparameters used for the different explainability techniques and Fashion-MNIST target classes used in my thesis.
If you want to run the respective training, make sure to replace all of the hyperparameters in the command above accordingly.
You should also specify the correct class ID: `included_classes=[5]` for Sandal and `included_classes=[4] ` for Coat.

|                        | Grad&#8209;CAM <br />x <br /> Sandal | Grad&#8209;CAM  <br /> x  <br /> Coat | Guided Backpropagation  <br /> x  <br /> Sandal | Guided Backpropagation  <br /> x  <br /> Coat |
|------------------------|:------------------------------------:|:-------------------------------------:|:-----------------------------------------------:|:---------------------------------------------:|
| lr                     | 0.00003                              | 0.00035                               | 0.00017                                         | 0.00006                                       |
| loss_weight_similarity | 1                                    | 1                                     | 2                                               | 1                                             |
| ce_class_weight        | 1                                    | 7                                     | 130                                             | 3                                             |
| batch_size             | 64                                   | 64                                    | 64                                              | 64                                            |
| weight_decay           | 0.00                                 | 0.00                                  | 0.00                                            | 0.001                                         |
| steps_lr               | 5                                    | 5                                     | 8                                               | 8                                             |
| gamma                  | 0.7                                  | 0.7                                   | 0.3                                             | 0.7                                           |
| max_epochs             | 30                                   | 30                                    | 30                                              | 30                                            |

The outputs of your training run will be saved at `logs/manipulate_model`.
The manipulated model checkpoints can then be found inside the run's `checkpoints` directory.

In order to perform a test run on your manipulated model, you can use the command
under the `Quick` section, adding the checkpoint of your trained model, as well as the correct `included_classes` and `explainer`.

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
