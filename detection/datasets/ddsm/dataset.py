from delira.data_loading import AbstractDataset
import pandas
from tqdm import tqdm
from detection.datasets.ddsm import ddsm_utils

class LazyDDSMDataset(AbstractDataset):
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

        if "csv_file" in self._load_kwargs:
            csv_file = self._load_kwargs["csv_file"]
        elif type == "mass":
            csv_file = '/home/temp/moriz/data/all_mass_cases.csv'
        else:
            csv_file = '/home/temp/moriz/data/all_calc_cases.csv'

        # two options supported: either a path to the directory with all
        # images or a list with the paths to all images
        if "path_list" in self._load_kwargs and \
            self._load_kwargs["path_list"] is not None:
            path_list = self._load_kwargs["path_list"]
        else:
            path_list = ddsm_utils.get_paths(self.data_path, csv_file)

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
        # data_dict = self._load_fn(self.data[index],
        #                           **self._load_kwargs)

        data_dict = self._load_fn(self.data[index], self.data_path,
                                  **self._load_kwargs)

        return data_dict

class CacheDDSMDataset(AbstractDataset):
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

        if "csv_file" in self._load_kwargs:
            csv_file = self._load_kwargs["csv_file"]
        elif type == "mass":
            csv_file = '/home/temp/moriz/data/mass_case_description_train_set.csv'
        else:
            csv_file = '/home/temp/moriz/data/calc_case_description_train_set.csv'

        # two options supported: either a path to the directory with all
        # images or a list with the paths to all images
        if "path_list" in self._load_kwargs and \
            self._load_kwargs["path_list"] is not None:
            path_list = self._load_kwargs["path_list"]
        else:
            path_list = ddsm_utils.get_paths(self.data_path, csv_file)

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
                                           self.data_path,
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

class OldCacheDDSMDataset(AbstractDataset):
    """
    Dataset to preload and cache data (data needs to fit in RAM!)
    """
    def __init__(self, data_path, csv_file, load_fn, **load_kwargs):
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
        super().__init__(data_path, load_fn, None, None)
        self._load_kwargs = load_kwargs
        self.csv_file = csv_file
        self.data = self._make_dataset(self.data_path, self.csv_file)


    def _make_dataset(self, data_path, csv_file):
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
        samples: list
            list of sample paths
        """
        # define data list
        data = []

        # read csv file using pandas
        data_frame = pandas.read_csv(csv_file)

        # define mammogram list to keep track of what was already considered
        mammogram_list = []

        # define how many data shall be loaded
        total_num_elements = data_frame.shape[0]
        if 'num_elements' not in self._load_kwargs \
                or self._load_kwargs['num_elements'] is None:
            num_elements = total_num_elements
        elif not isinstance(self._load_kwargs['num_elements'], int):
            TypeError("Number elements must be an int!")
        elif self._load_kwargs['num_elements'] > total_num_elements:
            ValueError("Chosen number elements must be smaller than the total "
                       "number of elements ({0})".format(total_num_elements))
        else:
            num_elements = self._load_kwargs['num_elements']

        for i in tqdm(range(num_elements)):
        #for i in tqdm(range(10)):
            # create unambiguous mammogram key
            patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
            patient_view = data_frame.iloc[i]['image view']
            patient_laterality = data_frame.iloc[i]['left or right breast']
            key = patient_id + "_" + patient_view + "_" + patient_laterality

            # check if mammogram already considered, continue if so
            if key in mammogram_list:
                continue

            # otherwise add key to list and load data
            mammogram_list.append(key)
            file_path = data_frame.iloc[i]['image file path']
            data.append(self._load_fn(file_path, self.csv_file,
                                      self.data_path, **self._load_kwargs))

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

class OldLazyDDSMDataset(AbstractDataset):
    """
    Dataset to load data in a lazy way
    """
    def __init__(self, data_path, csv_file, load_fn, **load_kwargs):
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
        self.csv_file = csv_file
        self.data = self._make_dataset(self.csv_file)


    def _make_dataset(self, csv_file):
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

        # define data list
        mammogram_paths = []

        # read csv file using pandas
        data_frame = pandas.read_csv(csv_file)

        # define mammogram list to keep track of what was already considered
        mammogram_list = []

        for i in tqdm(range(data_frame.shape[0])):
            # create unambiguous mammogram key
            patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
            patient_view = data_frame.iloc[i]['image view']
            patient_laterality = data_frame.iloc[i]['left or right breast']
            key = patient_id + "_" + patient_view + "_" + patient_laterality

            # check if mammogram already considered, continue if so
            if key in mammogram_list:
                continue

            # otherwise add key to list and load data
            mammogram_list.append(key)
            file_path = data_frame.iloc[i]['image file path']
            mammogram_paths.append(file_path)

        return mammogram_paths

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
        data_dict = self._load_fn(self.data[index], self.csv_file,
                                  self.data_path, **self._load_kwargs)

        return data_dict