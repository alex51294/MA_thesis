import sys
import torch
import argparse
import numpy as np
import yaml

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# my modules
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
from detection.datasets.ddsm.dataset import CacheDDSMDataset, LazyDDSMDataset
from detection.datasets.inbreast.dataset import LazyINbreastDataset, \
    CacheINbreastDataset
from detection.retinanet import RetinaNet
from detection.retinanet import losses
from detection.datasets.transforms import ConvertSegToBB
from detection.retinanet.experiment import RetinaNetExperiment
from detection.retinanet.config_handler import ConfigHandler

# delira modules
from delira.data_loading import BaseDataManager, \
    SequentialSampler, PrevalenceRandomSampler, RandomSampler, BaseDataLoader
from delira.training.train_utils import create_optims_default_pytorch
from delira.training import Parameters, PyTorchExperiment
from trixi.logger import PytorchVisdomLogger
from delira.logging import TrixiHandler
import logging
from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch

# framework for data augmentation; see link in delira for more information
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.sample_normalization_transforms import \
    MeanStdNormalizationTransform, ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, \
    SpatialTransform, Rot90Transform

def main(args):
    ########################################
    #                                      #
    #      DEFINE THE HYPERPARAMETERS      #
    #                                      #
    ########################################

    # load settings from config file
    config_file = args.get("config_file")
    config_handler = ConfigHandler()
    config_dict = config_handler(config_file)

    # some are changed rarely and given manually if required
    train_size = args.get("train_size")
    val_size = args.get("val_size")
    margin = args.get("margin")
    optimizer = args.get("optimizer")

    if optimizer != "SGD" and optimizer != "Adam":
        ValueError("Invalid optimizer")
    elif optimizer == "Adam":
        optimizer_cls = torch.optim.Adam
    else:
        optimizer_cls = torch.optim.SGD

    params = Parameters(
        fixed_params={"model": config_dict["model"],
                      "training": {**config_dict["training"],
                                   "optimizer_cls": optimizer_cls,
                                   **config_dict["optimizer"],
                                  "criterions": {"FocalLoss": losses.FocalLoss(),
                                                 "SmoothL1Loss": losses.SmoothL1Loss()},
                                  #  "criterions": {"FocalMSELoss": losses.FocalMSELoss(),
                                  #                 "SmoothL1Loss": losses.SmoothL1Loss()},
                                  # "lr_sched_cls": ReduceLROnPlateauCallbackPyTorch,
                                  # "lr_sched_params": {"verbose": True},
                                   "lr_sched_cls": None,
                                   "lr_sched_params": {},
                                  "metrics": {}}})

    ########################################
    #                                      #
    #        DEFINE THE AUGMENTATIONS      #
    #                                      #
    ########################################

    my_transforms = []
    mirror_transform = MirrorTransform(axes=(1, 2))
    my_transforms.append(mirror_transform)
    crop_size = config_dict["data"]["crop_size"]
    img_shape = config_dict["data"]["img_shape"]
    shape_limit = config_dict["data"]["shape_limit"]

    if (crop_size is not None and crop_size[0] == crop_size[1]) or \
        (img_shape is not None and len(img_shape) > 1
            and img_shape[0] == img_shape[1]):
        rot_transform = Rot90Transform(axes=(0, 1), p_per_sample=0.5)
        my_transforms.append(rot_transform)
    else:
        rot_transform = Rot90Transform(axes=(0, 1), num_rot=(0, 2),
                                       p_per_sample=0.5)
        my_transforms.append(rot_transform)

    # apply a more extended augmentation (if desiered)
    if "ext_aug" in config_dict["data"].keys() and \
            config_dict["data"]["ext_aug"] is not None and \
            config_dict["data"]["ext_aug"]:

        if crop_size is not None:
            size = [crop_size[0] + 25, crop_size[1] + 25]
        elif img_shape is not None:
            size = [img_shape[0] + 5, img_shape[1] + 5]
        elif shape_limit is not None:
            size = [shape_limit[0] + 5, shape_limit[1] + 5]
        else:
            raise KeyError("Crop size or image shape requried!")

        if crop_size is not None:
            spatial_transforms = SpatialTransform([size[0]-25, size[1]-25],
                                                  np.asarray(size) // 2,
                                                  do_elastic_deform=False,
                                                  do_rotation=True,
                                                  angle_x=(0, 0.01 * np.pi),
                                                  do_scale=True,
                                                  scale=(0.9, 1.1),
                                                  random_crop=True,
                                                  border_mode_data="mirror",
                                                  border_mode_seg="mirror")
            my_transforms.append(spatial_transforms)

        elif img_shape is not None or shape_limit is not None:
            spatial_transforms = SpatialTransform([size[0] - 5, size[1] - 5],
                                                  np.asarray(size) // 2,
                                                  do_elastic_deform=False,
                                                  do_rotation=False,
                                                  #angle_x=(0, 0.01 * np.pi),
                                                  do_scale=True,
                                                  scale=(0.9, 1.1),
                                                  random_crop=True,
                                                  border_mode_data="constant",
                                                  border_mode_seg="nearest")
            my_transforms.append(spatial_transforms)

    # bbox generation
    bb_transform = ConvertSegToBB(dim=2, margin=margin)
    my_transforms.append(bb_transform)

    transforms = Compose(my_transforms)

    ########################################
    #                                      #
    #   DEFINE THE DATASETS and MANAGER    #
    #                                      #
    ########################################

    # paths to csv files containing labels (and other information)
    csv_calc_train = '/home/temp/moriz/data/' \
                     'calc_case_description_train_set.csv'
    csv_mass_train = '/home/temp/moriz/data/' \
                     'mass_case_description_train_set.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'

    # determine class and load function
    if config_dict["data"]["dataset_type"] == "INbreast":
        dataset_cls = CacheINbreastDataset
        data_dir = inbreast_dir
        csv_file = None

        if config_dict["data"]["level"] == "crops":
            load_fn = inbreast_utils.load_pos_crops
        elif config_dict["data"]["level"] == "images":
            load_fn = inbreast_utils.load_sample
        elif config_dict["data"]["level"] == "both":
            #TODO: fix
            load_fn = inbreast_utils.load_sample_and_crops
        else:
            raise TypeError("Level required!")
    elif config_dict["data"]["dataset_type"] == "DDSM":
        data_dir = ddsm_dir

        if config_dict["data"]["level"] == "crops":
            load_fn = ddsm_utils.load_pos_crops
        elif config_dict["data"]["level"] == "images":
            load_fn = ddsm_utils.load_sample
        elif config_dict["data"]["level"] == "images+":
            load_fn = ddsm_utils.load_sample_with_crops
        else:
            raise TypeError("Level required!")

        if config_dict["data"]["type"] == "mass":
            csv_file = csv_mass_train
        elif config_dict["data"]["type"] == "calc":
            csv_file = csv_calc_train
        elif config_dict["data"]["type"] == "both":
            raise NotImplementedError("Todo")
        else:
            raise TypeError("Unknown lesion type!")

        if "mode" in config_dict["data"].keys():
            if config_dict["data"]["mode"] == "lazy":
                dataset_cls = LazyDDSMDataset

                if config_dict["data"]["level"] == "crops":
                    load_fn = ddsm_utils.load_single_pos_crops

            elif config_dict["data"]["mode"] == "cache":
                dataset_cls = CacheDDSMDataset
            else:
                raise TypeError("Unsupported loading mode!")
        else:
            dataset_cls = CacheDDSMDataset

    else:
        raise TypeError("Dataset is not supported!")

    dataset_train_dict = {'data_path': data_dir,
                          'xls_file': xls_file,
                          'csv_file': csv_file,
                          'load_fn': load_fn,
                          'num_elements': config_dict["debug"]["n_train"],
                          **config_dict["data"]}

    dataset_val_dict = {'data_path': data_dir,
                        'xls_file': xls_file,
                        'csv_file': csv_file,
                        'load_fn': load_fn,
                        'num_elements': config_dict["debug"]["n_val"],
                        **config_dict["data"]}

    datamgr_train_dict = {'batch_size': params.nested_get("batch_size"),
                          'n_process_augmentation': 4,
                          'transforms': transforms,
                          'sampler_cls': RandomSampler,
                          'data_loader_cls': BaseDataLoader}

    datamgr_val_dict = {'batch_size': params.nested_get("batch_size"),
                        'n_process_augmentation': 4,
                        'transforms': transforms,
                        'sampler_cls': SequentialSampler,
                        'data_loader_cls': BaseDataLoader}

    ########################################
    #                                      #
    #   INITIALIZE THE ACTUAL EXPERIMENT   #
    #                                      #
    ########################################
    checkpoint_path = config_dict["checkpoint_path"]["path"]
    # if "checkpoint_path" in args and args["checkpoint_path"] is not None:
    #     checkpoint_path = args.get("checkpoint_path")

    experiment = \
        RetinaNetExperiment(params,
                            RetinaNet,
                            name = config_dict["logging"]["name"],
                            save_path = checkpoint_path,
                            dataset_cls=dataset_cls,
                            dataset_train_kwargs=dataset_train_dict,
                            datamgr_train_kwargs=datamgr_train_dict,
                            dataset_val_kwargs=dataset_val_dict,
                            datamgr_val_kwargs=datamgr_val_dict,
                            optim_builder=create_optims_default_pytorch,
                            gpu_ids=list(range(args.get('gpus'))),
                            val_score_key="val_FocalLoss",
                            val_score_mode="lowest",
                            checkpoint_freq=2)

    ########################################
    #                                      #
    # LOGGING DEFINITION AND CONFIGURATION #
    #                                      #
    ########################################

    logger_kwargs = config_dict["logging"]

    # setup initial logging
    log_file = os.path.join(experiment.save_path, 'logger.log')


    logging.basicConfig(level=logging.INFO,
                        handlers=[TrixiHandler(PytorchVisdomLogger,
                                               **config_dict["logging"]),
                                  logging.StreamHandler(),
                                  logging.FileHandler(log_file)])

    logger = logging.getLogger("RetinaNet Logger")

    with open(experiment.save_path + "/config.yml", 'w') as file:
        yaml.dump(config_dict, file)

    ########################################
    #                                      #
    #       LOAD PATHS AND EXECUTE MODEL   #
    #                                      #
    ########################################
    seed = config_dict["data"]["seed"]

    if "train_size" in config_dict["data"].keys():
        train_size = config_dict["data"]["train_size"]

    if "val_size" in config_dict["data"].keys():
        val_size = config_dict["data"]["val_size"]

    if config_dict["data"]["dataset_type"] == "INbreast":
        if not config_dict["kfold"]["enable"]:


            train_paths, _, val_paths = \
                inbreast_utils.load_single_set(inbreast_dir,
                                               xls_file=xls_file,
                                               train_size=train_size,
                                               val_size=val_size,
                                               type=config_dict["data"]["type"],
                                               random_state=seed)

            if img_shape is not None or crop_size is not None:
                experiment.run(train_paths, val_paths)
            else:
                experiment.run(train_paths, None)

        else:
            paths = inbreast_utils.get_paths(inbreast_dir,
                                             xls_file=xls_file,
                                             type=config_dict["data"]["type"])

            if "splits" in config_dict["kfold"].keys():
                num_splits = config_dict["kfold"]["splits"]
            else:
                num_splits = 5

            experiment.kfold(paths,
                             num_splits=num_splits,
                             random_seed=seed,
                             dataset_type="INbreast")

    else:
        train_paths, val_paths, _ = \
            ddsm_utils.load_single_set(ddsm_dir,
                                       csv_file=csv_file,
                                       train_size=train_size,
                                       val_size=None,
                                       random_state=seed)

        if img_shape is not None or crop_size is not None:
            experiment.run(train_paths, val_paths)
        else:
            experiment.run(train_paths, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")

    #hyperparameter
    parser.add_argument('-e', '--epochs', required=False, type=int,
                        help='number of epochs',
                        default=5)
    parser.add_argument('-b', '--batchsize', required=False, type=int,
                        help='batchsize',
                        default=2)
    parser.add_argument('-lr', '--lr', required=False, type=float,
                        help='learning rate',
                        default=1e-4)
    parser.add_argument('-n_train', '--num_elements_train', required=False,
                        type=int,
                        help='num_elements_train',
                        default=None)
    parser.add_argument('-n_val', '--num_elements_val', required=False,
                        type=int,
                        help='num_elements_val',
                        default=None)
    parser.add_argument('-train_size', '--train_size', required=False,
                        type=float,
                        help='train_size',
                        default=0.9)
    parser.add_argument('-val_size', '--val_size', required=False, type=float,
                        help='val_size',
                        default=None)
    parser.add_argument('-img_shape', '--img_shape', nargs='+', required=False,
                        type=int,
                        help="img_shape",
                        default=None)
    parser.add_argument('-crop_size', '--crop_size', nargs='+', required=False,
                        type=int,
                        help="crop_size",
                        default=None)
    parser.add_argument('-shape_limit', '--shape_limit', nargs='+',
                        required=False,
                        type=int,
                        help="shape_limit",
                        default=None)
    parser.add_argument('-margin', '--margin', required=False, type=float,
                        help="margin",
                        default=0.1)
    parser.add_argument('-o', '--optimizer', required=False,
                        type=str,
                        help="optimizer",
                        default="Adam")
    parser.add_argument('-s', '--seed', required=False, type=int,
                        help='random_seed',
                        default=42)
    parser.add_argument('-n_outputs', '--num_outputs', required=False,
                        type=int,
                        help='num_outputs',
                        default=1)

    # experiment flags
    parser.add_argument('-ds_type', '--dataset_type', required=False,
                        type=str,
                        help="dataset_type",
                        default="INbreast")
    parser.add_argument('-l_type', '--lesion_type', required=False,
                        type=str,
                        help="lesion_type",
                        default="mass")
    parser.add_argument('-exp_name', '--exp_name', required=False,
                        type=str,
                        help="experiment_name",
                        default="test")
    parser.add_argument('-checkpoint_path', '--checkpoint_path',
                        required=False,
                        type=str,
                        help="checkpoint_path",
                        default="/home/temp/moriz/checkpoints/retinanet")
    parser.add_argument('-g', '--gpus', required=False, type=int,
                        help='number of gpus',
                        default=1)
    parser.add_argument('-vp', '--visdom_port', required=False, type=int,
                        help='visdom port',
                        default=9999)
    parser.add_argument('-vs', '--visdom_server', required=False, type=str,
                        help="visdom server",
                        default='http://pc87')
    parser.add_argument('-vpre', '--visdom_prefix', required=False,
                        type=str,
                        help="visdom_prefix",
                        default="test")

    # config file
    parser.add_argument('-config', '--config_file',
                        required=False,
                        type=str,
                        help="config_file",
                        default="/home/students/moriz/MA_Moriz/scripts/config.yml")

    args = vars(parser.parse_args())
    main(args)
