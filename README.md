# __Semi-Supervised Variational Adversarial Active Learning via Learning to Rank and Agreement-Based Pseudo Labeling__

__*Zongyao Lyu, William Beksi*__

Official Pytorch implementation for our ICPR 2024 paper.


# Abstract
_Active learning aims to alleviate the amount of labor involved in data labeling
by automating the selection of unlabeled samples via an acquisition function.
For example, variational adversarial active learning (VAAL) leverages an
adversarial network to discriminate unlabeled samples from labeled ones using
latent space information. However, VAAL has the following shortcomings: (i) it
does not exploit target task information, and (ii) unlabeled data is only used
for sample selection rather than model training. To address these limitations,
we introduce novel techniques that significantly improve the use of abundant
unlabeled data during training and take into account the task information.
Concretely, we propose an improved pseudo-labeling algorithm that leverages
information from all unlabeled data in a semi-supervised manner, thus allowing a
model to explore a richer data space. In addition, our method includes a
ranking-based loss prediction module that converts predicted relative ranking
information into a differentiable ranking loss. This loss can be embedded as a
rank variable into the latent space of a variational autoencoder and then
trained with a discriminator in an adversarial fashion for sample selection. We
demonstrate the superior performance of our approach over the state of the art
on various image classification and segmentation benchmark datasets._
## Prerequisites:   


## Requirements


## Running code

