from delira.data_loading import AbstractDataset
import pandas
import os
import numpy as np
from tqdm import tqdm

from detection.datasets.inbreast import inbreast_utils

class LazyINbreastDataset(AbstractDataset):
    """
    Dataset to load data in a lazy way
    """
    def __init__(self, data_path, load_fn, **load_kwargs):
        """

        Parameters
        ----------
        data_path: string
            path to data samples
        load_fn: function
            function to load single data sample
        csv_file:
            file containing all information according to the ddsm dataset
        load_kwargs: dict
            additional loading keyword arguments (image shape,
            channel number, ...); passed to load_fn
        """
        super().__init__(data_path, load_fn, '', '')
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset()


    def _make_dataset(self):
        """
        Helper Function to make a dataset containing paths to all images in a
        certain directory

        Parameters
        ----------
        data_path: string
            path to data samples

        csv_file: string
            path to csv file

        Returns
        -------
        mammogram_paths: list
            list of paths to mammograms
        """
        if "type" in self._load_kwargs:
            type = self._load_kwargs["type"]
        else:
            type = "mass"

        if "xls_file" in self._load_kwargs:
            xls_file = self._load_kwargs["xls_file"]
        else:
            xls_file = '/images/Mammography/INbreast/INbreast.xls'

        # two options supported: either a path to the directory with all
        # images or a list with the paths to all images
        if "path_list" in self._load_kwargs and \
                self._load_kwargs["path_list"] is not None:
            path_list = self._load_kwargs["path_list"]
        else:
            path_list = inbreast_utils.get_paths(self.data_path,
                                                 xls_file,
                                                 type)

        # define offset (useful for testing)
        if 'offset' in self._load_kwargs and \
                self._load_kwargs['offset'] is not None and \
                self._load_kwargs['offset'] < len(path_list):
            offset = self._load_kwargs['offset']
        else:
            offset = 0

        # define how many data shall be loaded
        if 'num_elements' in self._load_kwargs and \
                self._load_kwargs['num_elements'] is not None and \
                self._load_kwargs['num_elements'] < len(path_list):
            num_elements = self._load_kwargs['num_elements']
        else:
            num_elements = len(path_list)

        return path_list[offset:min(num_elements+offset, len(path_list))]

    def __getitem__(self, index):
        """
        load data sample specified by index

        Parameters
        ----------
        index: int
            index to specifiy which data sample to load

        Returns
        -------
        loaded data sample
        """
        data_dict = self._load_fn(self.data[index],
                                  **self._load_kwargs)

        return data_dict

class CacheINbreastDataset(AbstractDataset):
    """
    Dataset to load data in cache
    """
    def __init__(self, data_path, load_fn, **load_kwargs):
        """

        Parameters
        ----------
        data_path: string
            path to data samples
        load_fn: function
            function to load single data sample
        xls_file:
            file containing all information according to the inbreast dataset
        load_kwargs: dict
            additional loading keyword arguments (image shape,
            channel number, ...); passed to load_fn
        """
        super().__init__(data_path, load_fn, '', '')
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset()


    def _make_dataset(self):
        """
        Helper Function to make a dataset containing paths to all images in a
        certain directory

        Parameters
        ----------
        data_path: string
            path to data samples

        xls_file: string
            path to xls file

        Returns
        -------
        mammogram_paths: list
            list of paths to mammograms
        """
        if "type" in self._load_kwargs:
            type = self._load_kwargs["type"]
        else:
            type = "mass"

        if "xls_file" in self._load_kwargs:
            xls_file = self._load_kwargs["xls_file"]
        else:
            xls_file = '/images/Mammography/INbreast/INbreast.xls'

        # two options supported: either a path to the directory with all
        # images or a list with the paths to all images
        if "path_list" in self._load_kwargs and \
                self._load_kwargs["path_list"] is not None:
            path_list = self._load_kwargs["path_list"]
        else:
            path_list = inbreast_utils.get_paths(self.data_path,
                                                 xls_file,
                                                 type)

        # define offset (useful for testing)
        if 'offset' in self._load_kwargs and \
                self._load_kwargs['offset'] is not None and \
                self._load_kwargs['offset'] < len(path_list):
            offset = self._load_kwargs['offset']
        else:
            offset = 0

        # define how many data shall be loaded
        if 'num_elements' in self._load_kwargs and \
                self._load_kwargs['num_elements'] is not None and \
                self._load_kwargs['num_elements'] < len(path_list):
            num_elements = self._load_kwargs['num_elements']
        else:
            num_elements = len(path_list)

        path_list = path_list[offset:min(num_elements+offset, len(path_list))]

        # define data list
        data = []

        # load the desired data
        for i in tqdm(range(len(path_list))):
            # load data
            data_to_append = self._load_fn(path_list[i],
                                           **self._load_kwargs)

            # support two options: the general one, loading a dict of the form
            # {"data":data, "seg": seg} and a modified one, loading a list of
            # crops that is generated from one image, where each element
            # consist of a dict of the above form
            if isinstance(data_to_append, dict):
                data.append(data_to_append)
            elif isinstance(data_to_append, list):
                data = data + data_to_append

        return data

    def __getitem__(self, index):
        """
        load data sample specified by index

        Parameters
        ----------
        index: int
            index to specifiy which data sample to load

        Returns
        -------
        loaded data sample
        """
        data_dict = self.data[index]

        return data_dict