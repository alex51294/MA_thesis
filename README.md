# MA_thesis
The code for my thesis is built as following:
The package datasets contains the INbreast and the DDSM loader as well as two
files that contain necessary functions for both
(such as segmentation or augmentation).
The retinanet package contains all required parts for the model, with the
ResNet and FPN parts being saved in models and the RetinaNet specific parts
being implemented in retinanet (i.e. prepair batch, foward, closore...).
The anchors file contain the definition and generation function for anchors.
The losses contain the employed FocalLoss implementation as well as some work
in progress that did not work (yet).
The experiment is basically a wrapper for easier handling.
The retina_utils file contain some utils such as NMS for anchor merging.

Further, the scripts folder contains the scripts for training, validation and
evaluation as well as all the stuff around it.
The most important scripts are train (for obvious reasons), cb_validation,
cb_evaluation, wi_validation, wi_evaluation and plot_validation.
The validation is split in two parts: First, the x_validation files generates
for ALL considered epochs, applying anchor merging only and pickle the
respective results. The plot_validation script un-pickles them, applies a
second level of merging (if required) and plots the respective results.
The x_evaluation script evaluations for ONE model the performance on the test set.
For simplicity, the paths to the single experiment folders are saved in the
paths script, which is basically a simple getter function.
