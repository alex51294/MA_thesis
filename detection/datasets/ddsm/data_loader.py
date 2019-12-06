import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from multiprocessing import Pool
from delira.data_loading import AbstractDataset
from delira.data_loading.sampler import AbstractSampler, SequentialSampler


class BaseDataLoader(SlimDataLoaderBase):
    """
    Class to create a data batch out of data samples
    """
    def __init__(self, dataset: AbstractDataset,
                 batch_size=1, num_batches=None, seed=1,
                 sampler=None, n_process=1):
        """

        Parameters
        ----------
        dataset: subclass of AbstractDataset
            dataset to perform sample loading
        batch_size: int
            number of samples per batch
        num_batches: int
            number of batches to load
        seed: int
            seed for Random Number Generator
        sampler: AbstractSampler or None
            class defining the sampling strategy; if None: SequentialSampler
            will be used
        n_process: int or None
            number of processes for parallel loading. If None: process number
            equals number of CPUs
        """

        # store dataset in self._data
        super().__init__(dataset, batch_size)

        assert isinstance(sampler, AbstractSampler) or sampler is None, \
            "Sampler must be instance of subclass of AbstractSampler of None"

        if sampler is None:
            sampler = SequentialSampler(list(range(len(dataset))))

        self.sampler = sampler

        self.n_samples = len(dataset)
        if num_batches is None:
            num_batches = len(dataset) // batch_size

        self.num_batches = num_batches
        self._seed = seed
        np.random.seed(seed)

        self._batches_generated = 0
        self.n_process = n_process

    def generate_train_batch(self):
        """
        Generate Indices which behavior based on self.sampling gets data based
        on indices

        Returns
        -------
        dict: data and labels
        """

        if self._batches_generated >= self.num_batches:
            raise StopIteration
        else:
            self._batches_generated += 1

            idxs = self.sampler(self.batch_size)

            # with Pool(self.n_process) as p:
            #     result = p.map(self._get_sample, idxs)
            result = [self._get_sample(_idx) for _idx in idxs]

            result_dict = {}

            # concatenate dict entities by keys
            for _result_dict in result:
                for key, val in _result_dict.items():
                    if key in result_dict.keys():
                        result_dict[key].append(val)
                    else:
                        result_dict[key] = [val]

            # ATTENTION: only modification that is required (according to Justus)
            # convert list to numpy arrays
            # for key, val_list in result_dict.items():
            #     result_dict[key] = np.asarray(val_list)

            return result_dict

    def _get_sample(self, index):
        """
        Helper functions which returns an element of the dataset

        Parameters
        ----------
        index

        Returns
        -------

        """
        return self._data[index]

