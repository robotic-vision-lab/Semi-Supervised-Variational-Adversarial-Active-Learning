[//]: # (# __Semi-Supervised Variational Adversarial Active Learning via Learning to Rank and Agreement-Based Pseudo Labeling__)

[//]: # ()
[//]: # (### Overview)

[//]: # (_Active learning aims to alleviate the amount of labor involved in data labeling)

[//]: # (by automating the selection of unlabeled samples via an acquisition function.)

[//]: # (For example, variational adversarial active learning &#40;VAAL&#41; leverages an)

[//]: # (adversarial network to discriminate unlabeled samples from labeled ones using)

[//]: # (latent space information. However, VAAL has the following shortcomings: &#40;i&#41; it)

[//]: # (does not exploit target task information, and &#40;ii&#41; unlabeled data is only used)

[//]: # (for sample selection rather than model training. To address these limitations,)

[//]: # (we introduce novel techniques that significantly improve the use of abundant)

[//]: # (unlabeled data during training and take into account the task information.)

[//]: # (Concretely, we propose an improved pseudo-labeling algorithm that leverages)

[//]: # (information from all unlabeled data in a semi-supervised manner, thus allowing a)

[//]: # (model to explore a richer data space. In addition, our method includes a)

[//]: # (ranking-based loss prediction module that converts predicted relative ranking)

[//]: # (information into a differentiable ranking loss. This loss can be embedded as a)

[//]: # (rank variable into the latent space of a variational autoencoder and then)

[//]: # (trained with a discriminator in an adversarial fashion for sample selection. We)

[//]: # (demonstrate the superior performance of our approach over the state of the art)

[//]: # (on various image classification and segmentation benchmark datasets._)


## Semi-Supervised Variational Adversarial Active Learning via Learning to Rank and Agreement-Based Pseudo Labeling

### Overview

Active learning aims to alleviate the amount of labor involved in data labeling
by automating the selection of unlabeled samples via an acquisition function.
For example, variational adversarial active learning (VAAL) leverages an
adversarial network to discriminate unlabeled samples from labeled ones using
latent space information. However, VAAL has the following shortcomings: (i) it
does not exploit target task information, and (ii) unlabeled data is only used
for sample selection rather than model training. To address these limitations,
we introduce novel techniques that significantly improve the use of abundant
unlabeled data during training and take into account the task information.

<p align="center">
<img src="./misc/overview.png">
</p>

This repository provides source code for our 2024 ICPR paper titled "[Semi-Supervised
Variational Adversarial Active Learning via Learning to Rank and Agreement-Based
Pseudo Labeling]." 
Concretely, we propose an improved pseudo-labeling algorithm that leverages
information from all unlabeled data in a semi-supervised manner, thus allowing a
model to explore a richer data space. In addition, our method includes a
ranking-based loss prediction module that converts predicted relative ranking
information into a differentiable ranking loss. This loss can be embedded as a
rank variable into the latent space of a variational autoencoder and then
trained with a discriminator in an adversarial fashion for sample selection. We
demonstrate the superior performance of our approach over the state of the art
on various image classification and segmentation benchmark datasets.

### Citation

If you find this project useful, then please consider citing our work.

```
@inproceedings{lyu2024semi,
  title={Semi-Supervised Variational Adversarial Active Learning via Learning to
   Rank and Agreement-Based Pseudo Labeling},
  author={Lyu, Zongyao and Beksi, William J},
  booktitle={Proceedings of the International Conference on Pattern Recognition},
  year={2024}
}
```

[//]: # (### Model Architecture)

[//]: # ()
[//]: # (<p align="center">)

[//]: # (<img src="./misc/model.png">)

[//]: # (</p>)

### Installation

#### Prerequisites

Run the following command to install required packages.

```shell
  pip install -r requirements.txt 
```

### Usage

[//]: # (It's easy to run our code after you successfully install MMDetection. In order)

[//]: # (to make it more efficient, we separate the process of producing bounding boxes)

[//]: # (&#40;and their corresponding labels&#41; and performing inference. To produce boxes and)

[//]: # (labels, just set ``reuseFiles = False`` and ``saveFiles = True``. Then, run the)

[//]: # (following script)

[//]: # ()
[//]: # (```shell)

[//]: # (  python tools/testPOD.py )

[//]: # (```)

[//]: # ()
[//]: # (The boxes and labels file will be saved as a pickle file in the ``savedOutputs``)

[//]: # (folder. To do inference with an existing boxes and labels file, simply set)

[//]: # (``reuseFiles = True`` and ``saveFiles = False``. There are four primary)

[//]: # (hyperparameters we tuned in our work.)

[//]: # ()
[//]: # (- thresholds: Threshold for detection score)

[//]: # (- covPercents: Percent in which the covariance is scaled)

[//]: # (- boxratios: Ratio by which the bounding box is reduced)

[//]: # (- iouthresholds: IoU threshold)

[//]: # ()
[//]: # (Given each pair of produced boxes and labels, you can run inference using a)

[//]: # (combination of different values of hyperparameters. After you set the values of)

[//]: # (these parameters to the ones you want to test, just run the above script again.)

[//]: # (The computed PDQ score will be displayed in the terminal after inference)

[//]: # (finishes.)

### License 

[//]: # ([![license]&#40;https://img.shields.io/badge/license-Apache%202-blue&#41;]&#40;https://github.com/robotic-vision-lab/Deep-Ensembles-For-Probabilistic-Object-Detection/blob/main/LICENSE&#41;)


