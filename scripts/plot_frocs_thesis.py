import torch
import matplotlib
#matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import os
import time
import pickle

import scripts.plot_utils as plot_utils

# lr=1-e5, bs=1, 1800x900
# paths = {"RN152": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-18_20-26-01/run_00/checkpoint_best",
#          "RN101": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-18_19-28-15/run_00/checkpoint_best",
#          "RN50": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_16-28-02/run_00/checkpoint_best",
#          "RN34": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-29-55/run_00/checkpoint_best",
#          "RN18": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-31-43/run_00/checkpoint_best",
#          }

# lr=1-e5, bs=1, 1800x900
# paths = {"RN152": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-18_20-26-01/run_00/checkpoint_epoch_50",
#          "RN101": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-18_19-28-15/run_00/checkpoint_epoch_10",
#          "RN50": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_16-28-02/run_00/checkpoint_epoch_15",
#          "RN34": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-29-55/run_00/checkpoint_epoch_35",
#          "RN18": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-31-43/run_00/checkpoint_epoch_15",
#          }

#----------------------------------------------------------------------------

# lr=1-e5, bs=1, 1300x650
# paths = {"RN152": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-19_13-58-22/run_00/checkpoint_epoch_25",
#          "RN101": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-19_14-00-07/run_00/checkpoint_epoch_15",
#          "RN50": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_16-47-31/run_00/checkpoint_epoch_15",
#          "RN34": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-35-16/run_00/checkpoint_epoch_15",
#          "RN18": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-33-44/run_00/checkpoint_epoch_20",
#          }

# lr=1-e5, bs=1, 1300x650
# paths = {"RN152": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-19_13-58-22/run_00/checkpoint_best",
#          "RN101": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-19_14-00-07/run_00/checkpoint_best",
#          "RN50": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_16-47-31/run_00/checkpoint_best",
#          "RN34": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-35-16/run_00/checkpoint_best",
#          "RN18": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-18_20-33-44/run_00/checkpoint_best",
#          }


#----------------------------------------------------------------------------

# bs=1, lr=1e-5, 1300x650
# paths = {"RN50": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_21-48-21/run_00/checkpoint_best",
#          "RN34": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_22-03-11/run_00/checkpoint_best",
#          "RN18": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_22-04-44/run_00/checkpoint_best",
#          }

# bs=1, lr=1e-5, 1300x650
# paths = {"RN50": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_21-48-21/run_00/checkpoint_epoch_10",
#          "RN34": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_22-03-11/run_00/checkpoint_epoch_18",
#          "RN18": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_22-04-44/run_00/checkpoint_epoch_18",
#          }

#-----------------------------------------------------------------------------

# RN50, IN, 2600x1300, bs=1, different lr
# paths = {
#         "lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-17_16-03-15/run_00/checkpoint_best",
#          "lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-17_16-05-50/run_00/checkpoint_best",
#          "lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-17_16-06-24/run_00/checkpoint_best",
#          }

# paths = {
#         "lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-17_16-03-15/run_00/checkpoint_epoch_45",
#          "lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-17_16-05-50/run_00/checkpoint_epoch_25",
#          "lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-17_16-06-24/run_00/checkpoint_epoch_25",
#          }

#----------------------------------------------------------------------------

# RN50, IN, 1800x900, different lr, different bs
# paths = {
#         "bs_1_lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-17_21-41-08/run_00/checkpoint_best",
#         "bs_1_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-18_16-28-02/run_00/checkpoint_best",
#         "bs_1_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_20-57-44/run_00/checkpoint_best",
#         "bs_2_lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_21-11-42/run_00/checkpoint_best",
#         "bs_2_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_19-28-56/run_00/checkpoint_best",
#         "bs_2_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_21-28-12/run_00/checkpoint_best",
#          }

#---------------------------------------------------------------------------

# RN50, IN, 1300x650, different lr, different bs
# paths = {
#         "bs_1_lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-17_21-23-22/run_00/checkpoint_best",
#          "bs_1_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-18_16-47-31/run_00/checkpoint_best",
#          "bs_1_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_20-59-20/run_00/checkpoint_best",
#          "bs_2_lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-19_21-56-40/run_00/checkpoint_best",
#          "bs_2_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_21-01-16/run_00/checkpoint_best",
#          "bs_2_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_21-34-14/run_00/checkpoint_best",
#          "bs_4_lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_20-55-33/run_00/checkpoint_best",
#          "bs_4_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_19-30-50/run_00/checkpoint_best",
#          "bs_4_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-20_21-35-56/run_00/checkpoint_best",
#          }

#----------------------------------------------------------------------------

# RN50, IN, bs=1, lr=1e-5, size comparison
# paths = {"2600x1300": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-17_16-05-50/run_00/checkpoint_best",
#            "1800x900": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-18_16-28-02/run_00/checkpoint_best",
#            "1300x650": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-18_16-47-31/run_00/checkpoint_best",
#            }

# RN50, IN, bs=1, lr=1e-5, size comparison
paths = {"2600x1300": "/home/temp/moriz/validation/final/test_results/"
                            "19-04-17_16-05-50/run_00/checkpoint_epoch_25",
           "1800x900": "/home/temp/moriz/validation/final/test_results/"
                           "19-04-18_16-28-02/run_00/"
                       "checkpoint_epoch_15/IoU_0.5/",
           "1300x650": "/home/temp/moriz/validation/final/test_results/"
                           "19-04-18_16-47-31/run_00/checkpoint_epoch_15",
           }

#----------------------------------------------------------------------------


# # RN50, IN, bs=1, 2600x1300, segmented vs unsegmented
# paths = {"lr_1e-4_unseg": "/home/temp/moriz/validation/final/test_results/"
#                           "19-04-17_16-03-15/run_00/checkpoint_best",
#          "lr_1e-5_unseg": "/home/temp/moriz/validation/final/test_results/"
#                           "19-04-17_16-05-50/run_00/checkpoint_best",
#          "lr_1e-6_unseg": "/home/temp/moriz/validation/final/test_results/"
#                           "19-04-17_16-06-24/run_00/checkpoint_best",
#          "lr_1e-4_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-17_21-36-36/run_00/checkpoint_best",
#          "lr_1e-5_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-21_20-54-42/run_00/checkpoint_best",
#          "lr_1e-6_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-21_20-59-01/run_00/checkpoint_best",
#         }

#-------------------------------------------------------------------------

# RN50, IN, bs=1, 1800x900, segmented vs unsegmented
# paths = {"lr_1e-4": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-17_21-41-08/run_00/checkpoint_best",
#         "lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-18_16-28-02/run_00/checkpoint_best",
#          "lr_1e-4_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-17_21-35-44/run_00/checkpoint_best",
#          "lr_1e-5_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-21_21-06-40/run_00/checkpoint_best",
#         }


#------------------------------------------------------------------------

# RN50, IN, bs=1, 1300x650, segmented vs unsegmented
# paths = {"lr_1e-4_unseg": "/home/temp/moriz/validation/final/test_results/"
#                          "19-04-17_21-23-22/run_00/checkpoint_best",
#          "lr_1e-5_unseg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-18_16-47-31/run_00/checkpoint_best",
#          "lr_1e-4_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-19_21-57-44/run_00/checkpoint_best",
#          "lr_1e-5_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-21_21-07-38/run_00/checkpoint_best",
#         }

#--------------------------------------------------------------------------

# RN50, IN; bs=1, lr=1e-5, 2600x1300, fixed vs flexible size as well,
# segmented/unsegemented
# flexible describes here the fixing of one side only
# paths = {"fixed_size_unseg": "/home/temp/moriz/validation/final/test_results/"
#                              "19-04-17_16-05-50/run_00/checkpoint_epoch_25",
#          "fixed_size_seg": "/home/temp/moriz/validation/final/test_results/"
#                                "19-04-21_20-54-42/run_00/checkpoint_epoch_18",
#          "flex_size_unseg": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#          "flex_size_seg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_15-32-24/run_00/checkpoint_epoch_10",
#         }

#---------------------------------------------------------------------------

# RN50, IN; bs=1, lr=1e-5, different flexible size , segmented/unsegemented
# flexible describes here the fixing of one side only
# paths = {
#         "2600x1300_unseg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         "2600x1300_seg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_15-32-24/run_00/checkpoint_epoch_10",
#         "1800x900_unseg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_21-41-07/run_00/checkpoint_epoch_14",
#         "1800x900_seg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_21-41-46/run_00/checkpoint_epoch_50",
#         "1300x650_unseg": "/home/temp/moriz/validation/final/test_results/"
#                           "19-04-22_21-58-07/run_00/checkpoint_epoch_8",
#         "1300x650_seg": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-22_21-59-35/run_00/checkpoint_epoch_24",
#          }

#---------------------------------------------------------------------------

# RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# flexible describes here the complete isotropic down-scaling limited by a
# GPU-specific factor found by trial-and-error
# paths = {"fixed_size_unseg": "/home/temp/moriz/validation/final/"
#                              "test_results/"
#                              "19-04-17_16-05-50/run_00/checkpoint_epoch_25",
#          "fixed_size_seg": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-21_20-54-42/run_00/checkpoint_epoch_18",
#          "one_side_flex_unseg": "/home/temp/moriz/validation/final/"
#                                 "test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#          "one_side_flex_seg": "/home/temp/moriz/validation/final/"
#                               "test_results/"
#                               "19-04-22_15-32-24/run_00/checkpoint_epoch_10",
#          "full_flex_unseg.": "/home/temp/moriz/validation/final/"
#                              "test_results/"
#                              "19-04-23_21-57-17/run_00/checkpoint_epoch_18",
#          "full_flex_seg.": "/home/temp/moriz/validation/final/"
#                            "test_results/"
#                            "19-04-23_22-00-54/run_00/checkpoint_epoch_8",
#          }

#--------------------------------------------------------------------------

# RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# flexible describes here the complete isotropic down-scaling limited by a
# GPU-specific factor found by trial-and-error
# paths = {
#         "1080Ti_full_flex_unseg.": "/home/temp/moriz/validation/"
#                                    "final/test_results/"
#                     "19-04-23_21-57-17/run_00/checkpoint_epoch_18",
#          "1080Ti_full_flex_seg.": "/home/temp/moriz/validation/"
#                                   "final/test_results/"
#                     "19-04-23_22-00-54/run_00/checkpoint_epoch_8",
#          "980Ti_full_flex_unseg.": "/home/temp/moriz/validation/"
#                                    "final/test_results/"
#                     "19-04-24_18-48-20/run_00/checkpoint_epoch_26",
#          "980Ti_full_flex_seg.": "/home/temp/moriz/validation/"
#                                  "final/test_results/"
#                     "19-04-24_18-49-34/run_00/checkpoint_epoch_12",
#          }

#----------------------------------------------------------------------------

# RN50, IN, bs=1, lr=1e-5, label comparison
# paths = {"2600x1300_normal": "/home/temp/moriz/validation/final/test_results/"
#                                "19-04-17_16-05-50/run_00/checkpoint_best",
#            "2600x1300_pathology": "/home/temp/moriz/validation/"
#                                   "final/test_results/"
#                                   "19-04-21_21-00-57/run_00/checkpoint_best",
#            "2600x1300_birads": "/home/temp/moriz/validation/"
#                                "final/test_results/"
#                                "19-04-21_21-01-42/run_00/checkpoint_best",
#          }

############################################################################
############################################################################

# RN50, crops, IN, bs=1, lr=1e-5, ResNet comparison, cached
# paths = {"ResNet18_cache": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-21_20-44-17/run_00/checkpoint_best",
#          "ResNet34_cache": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-24_12-29-08/run_00/checkpoint_best",
#          "ResNet50_cache": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-24_12-28-33/run_00/checkpoint_best",
#          }

# RN50, crops, IN, bs=1, lr=1e-5, ResNet comparison
# paths = {"ResNet18": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-21_20-44-17/run_00/checkpoint_epoch_18",
#          "ResNet34": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-24_12-29-08/run_00/checkpoint_epoch_16",
#          "ResNet50": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-24_12-28-33/run_00/checkpoint_epoch_24",
#          }

#-----------------------------------------------------------------------------

# RN50, crops, IN, bs=1, lr=1e-5, ResNet comparison, lazy
# paths = {"ResNet18": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-26_18-31-41/run_00/checkpoint_best",
#          "ResNet34": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-26-57/run_00/checkpoint_best",
#          "ResNet50": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-27-35/run_00/checkpoint_best",
#          }

# paths = {"ResNet18": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-26_18-31-41/run_00/checkpoint_epoch_24",
#          "ResNet34": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-26-57/run_00/checkpoint_epoch_28",
#          "ResNet50": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-27-35/run_00/checkpoint_epoch_32",
#          }

#---------------------------------------------------------------------------


# RN50, crops, IN, bs=1, lr=1e-5, 900x900, size comparison
# paths = {"600x600": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_12-59-19/run_00/checkpoint_best",
#          "900x900": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-24_12-28-33/run_00/checkpoint_best",
#          "1200x1200": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_13-02-56/run_00/checkpoint_best",
#          }

#-----------------------------------------------------------------------------

# RN50, crops, IN, lr=1e-5, 900x900, bs comparison
# paths = {"bs_1": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-19_13-01-54/run_00/checkpoint_best",
#          "bs_2": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-21_20-42-38/run_00/checkpoint_best",
#          "bs_4": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-21_20-41-55/run_00/checkpoint_best",
#          }

#----------------------------------------------------------------------------

# RN50, crops, IN, bs=1, lr=1e-5, size comparison
# paths = {"lr_1e-4_normal": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-22_21-55-18/run_00/checkpoint_best",
#         "lr_1e-5_normal": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-19_13-01-54/run_00/checkpoint_best",
#          "lr_1e-6_normal": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_21-52-34/run_00/checkpoint_best",
#         "lr_1e-4_better": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-21_21-16-26/run_00/checkpoint_best",
#         "lr_1e-5_better": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-22_21-54-47/run_00/checkpoint_best",
#          "lr_1e-6_better": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_21-53-52/run_00/checkpoint_best",
#          }

#----------------------------------------------------------------------------

# RN18, crops, lr=1e-5, 900x900, bs and norm comparison
# paths = {"IN_bs_1": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-31-41/run_00/checkpoint_best",
#          "IN_bs_2": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-36-27/run_00/checkpoint_best",
#          "IN_bs_4": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_19-30-10/run_00/checkpoint_best",
#          "BN_bs_4": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_19-02-14/run_00/checkpoint_best",
#          "BN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-56-05/run_00/checkpoint_best",
#          }

#---------------------------------------------------------------------------
# # RN18, crops, lr=1e-5, bs=1, crop size comparison
# paths = {"600x600": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-30-29/run_00/checkpoint_best",
#          "900x900": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-31-41/run_00/checkpoint_best",
#          "1200x1200": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-34-13/run_00/checkpoint_best",
#          }

# paths = {"600x600": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-30-29/run_00/checkpoint_epoch_24",
#          "900x900": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-31-41/run_00/checkpoint_epoch_24",
#          "1200x1200": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-34-13/run_00/checkpoint_epoch_48",
#          }

#--------------------------------------------------------------------------

# RN18, crops, lr=1e-5, bs=1, 900x900 mode comparison
# paths = {"cache": "/home/temp/moriz/validation/final/test_results/"
#                   "19-04-21_20-44-17/run_00/checkpoint_best",
#          "lazy": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-26_18-31-41/run_00/checkpoint_best",
#          }

#--------------------------------------------------------------------------

# RN18, crops, lr=1e-5 1200x1200, bs and norm comp
# paths = {"IN_bs_1": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-26_18-34-13/run_00/checkpoint_best",
#          "IN_bs_2": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-58-38/run_00/checkpoint_best",
#          "IN_bs_4": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-56-50/run_00/checkpoint_best",
#          "IN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-50-45/run_00/checkpoint_best",
#          "BN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_23-11-34/run_00/checkpoint_best",
#          }

# RN18, crops, lr=1e-5 1200x1200, bs and norm comp
# paths = {"IN_bs_1": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-26_18-34-13/run_00/checkpoint_epoch_20",
#          "IN_bs_2": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-58-38/run_00/checkpoint_best",
#          "IN_bs_4": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-56-50/run_00/checkpoint_epoch_16",
#          "IN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-50-45/run_00/checkpoint_epoch_24",
#          "BN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_23-11-34/run_00/checkpoint_epoch_32",
#          }

#--------------------------------------------------------------------------

# RN18, crops, lr=1e-6 1200x1200, bs and norm comp
# paths = {"IN_bs_1": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_17-01-16/run_00/checkpoint_best",
#          "IN_bs_2": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-59-57/run_00/checkpoint_best",
#          "IN_bs_4": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-55-51/run_00/checkpoint_best",
#          "IN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-52-23/run_00/checkpoint_best",
#          "BN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_23-15-01/run_00/checkpoint_best",
#          }

# paths = {"IN_bs_1": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_17-01-16/run_00/checkpoint_epoch_40",
#          "IN_bs_2": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-59-57/run_00/checkpoint_epoch_44",
#          "IN_bs_4": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-55-51/run_00/checkpoint_best",
#          "IN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_16-52-23/run_00/checkpoint_best",
#          "BN_bs_8": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-27_23-15-01/run_00/checkpoint_best",
#          }

#----------------------------------------------------------------------------

# paths = {"IN_bs_4_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-27_16-56-50/run_00/checkpoint_best",
#          "IN_bs_1_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-27_17-01-16/run_00/checkpoint_best",
#          }

# paths = {"IN_bs_4_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-27_16-56-50/run_00/checkpoint_epoch_16",
#          "IN_bs_1_lr_1e-6": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-27_17-01-16/run_00/checkpoint_epoch_40",
#          }

#--------------------------------------------------------------------------

# RN18, bs=1, lr=1e-5, lr=1e-6, better aug comp
# paths = {"IN_bs_1_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-26_18-34-13/run_00/checkpoint_best",
#          "IN_bs_1_lr_1e-5_better_aug": "/home/temp/moriz/validation/"
#                                        "final/test_results/"
#                             "19-04-27_23-00-22/run_00/checkpoint_best",
#          "IN_bs_1_lr_1e-6": "/home/temp/moriz/validation/"
#                             "final/test_results/"
#                             "19-04-27_17-01-16/run_00/checkpoint_best",
#          "IN_bs_1_lr_1e-6_better_aug": "/home/temp/moriz/validation/"
#                                        "final/test_results/"
#                             "19-04-27_23-01-58/run_00/checkpoint_best",
#          }

# paths = {"IN_bs_1_lr_1e-5": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-26_18-34-13/run_00/checkpoint_epoch_48",
#          "IN_bs_1_lr_1e-5_better_aug": "/home/temp/moriz/validation/"
#                                        "final/test_results/"
#                             "19-04-27_23-00-22/run_00/checkpoint_epoch_44",
#          "IN_bs_1_lr_1e-6": "/home/temp/moriz/validation/"
#                             "final/test_results/"
#                             "19-04-27_17-01-16/run_00/checkpoint_epoch_40",
#          "IN_bs_1_lr_1e-6_better_aug": "/home/temp/moriz/validation/"
#                                        "final/test_results/"
#                             "19-04-27_23-01-58/run_00/checkpoint_epoch_24",
#          }


##############################################################################
##############################################################################

# RN18, one stage vs. multistage comparison
# paths = {"one-stage": "/home/temp/moriz/validation/final/test_results/"
#                       "19-04-18_20-31-43/run_00/checkpoint_best",
#          "multi-stage": "/home/temp/moriz/validation/test/"
#                         "19-04-26_20-59-29/run_00/checkpoint_best",
#          }


# paths = {"1080Ti_full_flex_seg._HP": "/home/temp/moriz/validation/"
#                                      "final/test_results/"
#                                      "19-04-23_22-00-54/run_00/"
#                                      "checkpoint_epoch_8",
#          "980Ti_full_flex_seg._HP": "/home/temp/moriz/validation/"
#                                     "final/test_results/"
#                                     "19-04-24_18-49-34/run_00/"
#                                     "checkpoint_epoch_12",
#          "1080Ti_full_flex_seg._FP": "/home/temp/moriz/validation/"
#                                      "final/test_results/"
#                                   "19-04-28_13-54-55/run_00/"
#                                      "checkpoint_epoch_10",
#          "980Ti_full_flex_seg._FP": "/home/temp/moriz/validation/"
#                                     "final/test_results/"
#                                     "19-04-28_13-52-53/run_00/"
#                                     "checkpoint_epoch_12",
#          }


# paths = {"IoU=0.5, plain": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#          "IoU=0.5, enhanced": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         # "IoU=0.4, plain": "/home/temp/moriz/validation/final/test_results/"
#         #                      "19-04-22_15-32-16/run_00/checkpoint_epoch_12/"
#         #                      "IoU_0.4",
#         # "IoU=0.3, plain": "/home/temp/moriz/validation/final/test_results/"
#         #                      "19-04-22_15-32-16/run_00/checkpoint_epoch_12/"
#         #                      "IoU_0.3",
#          "IoU=0.2, plain": "/home/temp/moriz/validation/final/test_results/"
#                              "19-04-22_15-32-16/run_00/checkpoint_epoch_12/"
#                              "IoU_0.2",
#         # "IoU=0.1, plain": "/home/temp/moriz/validation/final/test_results/"
#         #                      "19-04-22_15-32-16/run_00/checkpoint_epoch_12/"
#         #                      "IoU_0.1",
#          "IoU=0.2, enhanced": "/home/temp/moriz/validation/final/test_results/"
#                              "19-04-22_15-32-16/run_00/checkpoint_epoch_12/"
#                              "IoU_0.2_enh",
#          }

# paths = {"normal aug.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         "better aug.": "/home/temp/moriz/validation/final/test_results/"
#                              "19-04-28_21-24-09/run_00/checkpoint_epoch_40/",
#          }

# paths = {"mass det.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         "calc. det.": "/home/temp/moriz/validation/final/test_results/"
#                              "19-04-29_09-47-38/run_00/checkpoint_epoch_32/",
#          }

# paths = {"DDSM": "/home/temp/moriz/validation/final/test_results/"
#                  "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         "INbreast": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-22_15-32-16/run_00/"
#                     "checkpoint_epoch_12/INbreast/",
#         "CL, 1e-5, DDSM": "/home/temp/moriz/validation/final/test_results/"
#                           "19-05-01_11-22-10/run_00/checkpoint_epoch_8/",
#          "CL, 1e-5 INbreast": "/home/temp/moriz/validation/final/test_results/"
#                               "19-05-01_11-22-10/run_00/"
#                               "checkpoint_epoch_8/INbreast/",
#          "CL, 1e-6, DDSM": "/home/temp/moriz/validation/final/test_results/"
#                            "19-05-01_11-23-58/run_00/"
#                            "checkpoint_epoch_26/",
#          "CL, 1e-6, INbreast": "/home/temp/moriz/validation/"
#                                "final/test_results/"
#                                "19-05-01_11-23-58/run_00/"
#                                "checkpoint_epoch_26/INbreast/",
#          }

# # RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# # flexible describes here the complete isotropic down-scaling limited by a
# paths = {"one_side_flex_unseg": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#          "one_side_flex_seg_A": "/home/temp/moriz/validation/final/test_results/"
#                               "19-04-22_15-32-24/run_00/checkpoint_epoch_10",
#          "one_side_flex_seg_B": "/home/temp/moriz/validation/final/test_results/"
#                               "19-04-30_20-28-21/run_00/checkpoint_epoch_26",
#
#          }


# # RN50, IN; bs=1, lr=1e-5, different flexible size , segmented/unsegemented
# # flexible describes here the fixing of one side only
# paths = {
#         "2600x1300, fixed size": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-17_16-05-50/run_00/checkpoint_epoch_25",
#          # "1800x900, fixed size": "/home/temp/moriz/validation/final/test_results/"
#          #                   "19-04-18_16-28-02/run_00/checkpoint_epoch_15",
#          # "1300x650, fixed size": "/home/temp/moriz/validation/final/test_results/"
#          #                   "19-04-18_16-47-31/run_00/checkpoint_epoch_15",
#         "2600x1300, shape limit": "/home/temp/moriz/validation/final/test_results/"
#                            "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         # "1800x900, shape limit": "/home/temp/moriz/validation/final/test_results/"
#         #                    "19-04-22_21-41-07/run_00/checkpoint_epoch_14",
#         # "1300x650, shape limit": "/home/temp/moriz/validation/final/test_results/"
#         #                   "19-04-22_21-58-07/run_00/checkpoint_epoch_8",
#         "2600x1300, area limit*": "/home/temp/moriz/validation/final/test_results/"
#                     "19-04-23_21-57-17/run_00/checkpoint_epoch_18",
#         # "1800x900, area limit*": "/home/temp/moriz/validation/final/test_results/"
#         #             "19-04-24_18-48-20/run_00/checkpoint_epoch_26",
#          }

#----------------------------------------------------------------------------

# paths = {"baseline, RN50": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12/IoU_0.2/",
#          "baseline, RN18": "/home/temp/moriz/validation/final/test_results/"
#                            "19-05-05_15-25-37/run_00/checkpoint_epoch_22/IoU_0.2/",
#          # "CL, RN50, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#          #                     "19-05-01_11-22-10/run_00/checkpoint_epoch_8/IoU_0.2/",
#          # "CL, RN50, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#          #                        "19-05-01_11-23-58/run_00/checkpoint_epoch_26/IoU_0.2/",
#          "CL, RN18, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#                              "19-05-01_11-25-14/run_00/checkpoint_epoch_28/IoU_0.2/",
#          "CL, RN18, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_11-25-55/run_00/checkpoint_epoch_28//IoU_0.2/",
#          }

# paths = {
#         "baseline, RN50": "/home/temp/moriz/validation/final/test_results/"
#                                     "19-04-22_15-32-16/run_00/checkpoint_epoch_12/INbreast_IoU_0.2/",
#         # "baseline, RN18": "/home/temp/moriz/validation/final/test_results/"
#         #                        "19-05-05_15-25-37/run_00/checkpoint_epoch_22/INbreast_IoU_0.2/",
#          "CL, RN50, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#                              "19-05-01_11-22-10/run_00/checkpoint_epoch_8/INbreast_IoU_0.2",
#          "CL, RN50, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_11-23-58/run_00/checkpoint_epoch_26/INbreast_IoU_0.2/",
#          # "CL, RN18, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#          #                     "19-05-01_11-25-14/run_00/checkpoint_epoch_28/INbreast_IoU_0.2/",
#          # "CL, RN18, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#          #                        "19-05-01_11-25-55/run_00/checkpoint_epoch_28/INbreast_IoU_0.2/",
#          }

#---------------------------------------------------------------------------

# paths = {
#         "baseline, RN50": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-17_16-05-50/run_00/checkpoint_epoch_25/IoU_0.2/",
#          # "baseline, RN18": "/home/temp/moriz/validation/final/test_results/"
#          #                   "19-05-05_16-55-51/run_00/checkpoint_epoch_22/IoU_0.2",
#          "CL, RN50, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#                              "19-05-05_16-51-09/run_00/checkpoint_epoch_12/IoU_0.2/",
#          "CL, RN50, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-05_16-52-08/run_00/checkpoint_epoch_14/IoU_0.2/",
#          # "CL, RN18, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#          #                     "19-05-05_16-43-00/run_00/checkpoint_epoch_18/IoU_0.2/",
#          # "CL, RN18, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#          #                        "19-05-05_16-47-51/run_00/checkpoint_epoch_26/IoU_0.2",
#          }

# paths = {
#         "baseline, RN50": "/home/temp/moriz/validation/final/test_results/"
#                           "19-04-17_16-05-50/run_00/"
#                           "checkpoint_epoch_25/INbreast_IoU_0.2/",
#          # "baseline, RN18": "/home/temp/moriz/validation/final/test_results/"
#          #                   "19-05-05_16-55-51/run_00/checkpoint_epoch_22/IoU_0.2",
#          "CL, RN50, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#                              "19-05-05_16-51-09/run_00/"
#                               "checkpoint_epoch_12/INbreast_IoU_0.2/",
#          "CL, RN50, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#                               "19-05-05_16-52-08/run_00/"
#                               "checkpoint_epoch_14/INbreast_IoU_0.2/",
#          # "CL, RN18, lr=1e-5": "/home/temp/moriz/validation/final/test_results/"
#          #                     "19-05-05_16-43-00/run_00/"
#          #                      "checkpoint_epoch_18/INbreast_IoU_0.2/",
#          # "CL, RN18, lr=1e-6": "/home/temp/moriz/validation/final/test_results/"
#          #                        "19-05-05_16-47-51/run_00/"
#          #                      "checkpoint_epoch_26/INbreast_IoU_0.2",
#          }



#-----------------------------------------------------------------------------

# # # RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# paths = {"2600x1300, fixed size, unseg.": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-17_16-05-50/run_00/checkpoint_epoch_25",
#         "2600x1300, fixed size, seg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-29-19/run_00/checkpoint_epoch_18",
#         "2600x1300, flex. size, unseg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         "2600x1300, flex. size, seg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-26-23/run_00/checkpoint_epoch_6",
#          }
#
# # RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# # flexible describes here the complete isotropic down-scaling limited by a
# paths_2 = {"1800x900, fixed size, unseg.": "/home/temp/moriz/validation/final/test_results/"
#                             "19-04-18_16-28-02/run_00/checkpoint_epoch_15",
#         "1800x900, fixed size, seg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-28-33/run_00/checkpoint_epoch_20",
#         "1800x900, flex. size, unseg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_21-41-07/run_00/checkpoint_epoch_14",
#         "1800x900, flex. size, seg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-27-06/run_00/checkpoint_epoch_8",
#          }

# # RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# paths = {"2600x1300, flex. size, unseg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12",
#         "2600x1300, flex. size, unseg., better aug.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-34-30/run_00/checkpoint_epoch_26",
#          "2600x1300, flex. size, seg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-26-23/run_00/checkpoint_epoch_6",
#         "2600x1300, flex. size, seg., better aug.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-32-48/run_00/checkpoint_epoch_16",
#          }

# RN50, IN, bs=1, lr=1e-5, max. flex size, seg/unseg
# paths = {"2600x1300, flex. size, unseg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-22_15-32-16/run_00/checkpoint_epoch_12/INbreast",
#         # "2600x1300, flex. size, unseg., better aug.": "/home/temp/moriz/validation/final/test_results/"
#         #                         "19-05-01_22-34-30/run_00/checkpoint_epoch_26/INbreast",
#          "2600x1300, flex. size, seg.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-05-01_22-26-23/run_00/checkpoint_epoch_6/INbreast",
#         # "2600x1300, flex. size, seg., better aug.": "/home/temp/moriz/validation/final/test_results/"
#         #                         "19-05-01_22-32-48/run_00/checkpoint_epoch_16/INbreast",
#          }

# paths = {
#         "mass det.": "/home/temp/moriz/validation/final/test_results/"
#                                 "19-04-27_16-56-50/run_00/checkpoint_epoch_16",
#         #  "mass det.": "/home/temp/moriz/validation/final/test_results/"
#         #                         "19-04-26_18-34-13/run_00/checkpoint_epoch_48",
#         # "calc. det.": "/home/temp/moriz/validation/final/test_results/"
#         #                      "19-05-03_00-10-53/run_00/checkpoint_epoch_20/",
#         "calc. det.": "/home/temp/moriz/validation/final/test_results/"
#                              "19-05-07_15-04-32/run_00/checkpoint_epoch_24/",
#          }

#----------------------------------------------------------------------------

# RN50, IN; bs=1, lr=1e-5, 2600x1300, INbreast-DDSM comparison
paths = {
        "INbreast-trained, fold 1": "/home/temp/moriz/validation/final"
                                    "/test_results/19-04-23_10-35-57/run_00/"
                                    "checkpoint_epoch_18/IoU_0.2/",
        "INbreast-trained, fold 2": "/home/temp/moriz/validation/final"
                                    "/test_results/19-04-23_10-35-57/run_01/"
                                    "checkpoint_epoch_20/IoU_0.2/",
        "INbreast-trained, fold 3": "/home/temp/moriz/validation/final"
                                "/test_results/19-04-23_10-35-57/run_02/"
                                "checkpoint_epoch_38/IoU_0.2/",
        "INbreast-trained, fold 4": "/home/temp/moriz/validation/final"
                                "/test_results/19-04-23_10-35-57/run_03/"
                                "checkpoint_epoch_46/IoU_0.2/",
        "INbreast-trained, fold 5": "/home/temp/moriz/validation/final"
                                "/test_results/19-04-23_10-35-57/run_04/"
                                "checkpoint_epoch_44/IoU_0.2/",
        # "DDSM-trained": "/home/temp/moriz/validation/final/test_results/"
        #                 "19-04-17_16-05-50/run_00/"
        #                 "checkpoint_epoch_25/INbreast_IoU_0.2",
        }

# paths = {
#         "INbreast-trained, fold 1": "/home/temp/moriz/validation/final"
#                                     "/test_results/19-04-29_09-57-24/run_00/"
#                                     "checkpoint_epoch_40/",
#         "INbreast-trained, fold 2": "/home/temp/moriz/validation/final"
#                                     "/test_results/19-04-29_09-57-24/run_01/"
#                                     "checkpoint_epoch_22/",
#         "INbreast-trained, fold 3": "/home/temp/moriz/validation/final"
#                                 "/test_results/19-04-29_09-57-24/run_02/"
#                                 "checkpoint_epoch_42/",
#         "INbreast-trained, fold 4": "/home/temp/moriz/validation/final"
#                                 "/test_results/19-04-29_09-57-24/run_03/"
#                                 "checkpoint_epoch_28/",
#         "INbreast-trained, fold 5": "/home/temp/moriz/validation/final"
#                                 "/test_results/19-04-29_09-57-24/run_04/"
#                                 "checkpoint_epoch_50/",
#         "DDSM-trained": "/home/temp/moriz/validation/final/test_results/"
#                         "19-04-27_16-56-50/run_00/"
#                         "checkpoint_epoch_16/INbreast/",
#         }


#paths = {**paths, **paths_2}

froc_tpr = []
froc_fppi = []

for key in paths.keys():
    # load settings
    with open(paths[key] + "/results", "rb") as results_file:
        results_dict = pickle.load(results_file)

    froc_tpr.append(results_dict["FROC"]["TPR"])
    froc_fppi.append(results_dict["FROC"]["FPPI"])

plot_utils.plot_frocs(froc_tpr, froc_fppi, left_range=1e-1, right_range=4,
                      legend=[key for key in paths.keys()],
                      legend_position=4,
                      ylim=(0.65, 1.01),
                      image_save=True,
                      ##image_save_dir="/home/temp/moriz/validation/final/frocs",
                      #image_save_dir="/home/temp/moriz/validation/final/plots_thesis",
                      image_save_dir="/home/students/moriz/MA_Moriz/thesis/images/plots",
                      #image_save_dir="/home/students/moriz/Bilder/test",
                      image_save_name="inbreast_kfolds",
                      )
#print(np.asarray(froc_tpr).shape)