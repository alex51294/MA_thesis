import os
import logging
import numpy as np
import torch

# delira modules
from delira.data_loading import BaseDataManager, AbstractDataset
from delira.training.train_utils import create_optims_default_pytorch
from delira.training import Parameters, PyTorchExperiment, \
    PyTorchNetworkTrainer as PTNetworkTrainer

from delira.models import AbstractPyTorchNetwork

import detection.datasets.utils as utils

import logging


logger = logging.getLogger(__name__)


class RetinaNetExperiment(PyTorchExperiment):
    """
    Single Experiment
    """

    def __init__(self,
                 params: Parameters,
                 model_cls: AbstractPyTorchNetwork,
                 dataset_cls: AbstractDataset,
                 dataset_train_kwargs: dict,
                 datamgr_train_kwargs: dict,
                 dataset_val_kwargs: dict = None,
                 datamgr_val_kwargs: dict = None,
                 name=None,
                 save_path=None,
                 val_score_key=None,
                 optim_builder=create_optims_default_pytorch,
                 checkpoint_freq=1,
                 trainer_cls=PTNetworkTrainer,
                 **kwargs):
        """

        Parameters
        ----------
        name: string
            Experiment name
        save_path: string
            path where all outputs and weights belonging to this experiment
            are saved
        hyper_params: Hyperparameters
            class containing all hyper parameters. Must provide attributes
            'metrics', 'criterions', 'optimizer_cls', 'optimizer_params',
            'lr_scheduler_cls', lr_scheduler_params'
        model_cls: AbstractNetwork
            actual model class
        model_kwargs: dict
            keyword arguments to create model instance
        val_score_key: string
            key specifying metric to decide for best model
        kwargs: dict
            additional keyword arguments
        """

        super().__init__(params,
                         model_cls,
                         name=name,
                         save_path=save_path,
                         val_score_key=val_score_key,
                         optim_builder=optim_builder,
                         checkpoint_freq=checkpoint_freq,
                         trainer_cls=trainer_cls,
                         **kwargs)

        self.dataset_cls = dataset_cls
        self.dataset_train_kwargs = dataset_train_kwargs
        self.datamgr_train_kwargs = datamgr_train_kwargs
        self.dataset_val_kwargs = dataset_val_kwargs
        self.datamgr_val_kwargs = datamgr_val_kwargs


    def run(self, train_paths, val_paths=None):

        dataset_train = self.dataset_cls(path_list=train_paths,
                                         **self.dataset_train_kwargs)
        mgr_train = BaseDataManager(dataset_train,
                                    **self.datamgr_train_kwargs)

        if val_paths is not None:
            dataset_val = self.dataset_cls(path_list=val_paths,
                                           **self.dataset_val_kwargs)

            mgr_val = BaseDataManager(dataset_val,
                                      **self.datamgr_train_kwargs)

        if val_paths is not None:
            super().run(mgr_train, mgr_val)
        else:
            super().run(mgr_train, None)


    def kfold(self, paths, num_splits=5, shuffle=True,
              random_seed=None, valid_size=0.1, **kwargs):
        """
        Runs K-Fold Crossvalidation

        Parameters
        ----------
        num_epochs: int
            number of epochs to train the model
        data: str
            path to root dir
        num_splits: None or int
            number of splits for kfold
            if None: len(data) splits will be validated
        shuffle: bool
            whether or not to shuffle indices for kfold
        random_seed: None or int
            random seed used to seed the kfold (if shuffle is true),
            pytorch and numpy
        valid_size : float, default: 0.1
            relative size of validation dataset in relation to training set
        """

        if random_seed is not None:
            torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        if "dataset_type" in kwargs and kwargs["dataset_type"] is not None:
            dataset_type = kwargs["dataset_type"]
            if dataset_type != "INbreast" and dataset_type != "DDSM":
                raise ValueError("Unknown dataset!")
        else:
            raise ValueError("No dataset type!")

        train_splits, _ = utils.kfold_patientwise(paths,
                                                  dataset_type=dataset_type,
                                                  num_splits=num_splits,
                                                  shuffle=shuffle,
                                                  random_state=random_seed)

        for i in range(len(train_splits)):
            train_paths, val_paths, _ = \
                utils.split_paths_patientwise(train_splits[i],
                                              dataset_type=dataset_type,
                                              train_size= 1 - valid_size)

            dataset_train = self.dataset_cls(path_list=train_paths,
                                             **self.dataset_train_kwargs)

            dataset_valid = self.dataset_cls(path_list=val_paths,
                                             **self.dataset_val_kwargs)

            mgr_train = BaseDataManager(dataset_train,
                                        **self.datamgr_train_kwargs)

            mgr_valid = BaseDataManager(dataset_valid,
                                        **self.datamgr_val_kwargs)

            super().run(mgr_train, mgr_valid,
                        fold=i,
                        **kwargs)