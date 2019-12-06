import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils

from detection.datasets import LazyDDSMDataset, CacheINbreastDataset, \
    LazyINbreastDataset

# maybe unnecessary, but did not work otherwise
backend = matplotlib.get_backend()
if backend != 'TkAgg':
    matplotlib.use('module://backend_interagg')

def main(dataset, image_save=False, image_save_dir="/home/temp/moriz"):

    label_dict = {"0.0": 0, "1.0": 0, "2.0": 0, "3.0": 0,
                  "4.0": 0, "5.0": 0, "6.0": 0}
    for i in tqdm(range(len(dataset))):
       label = dataset[i]["label"]
       label_dict[str(label)] += 1

    occurrence = []
    for key in label_dict.keys():
        occurrence.append(label_dict[key])
        #print("Number of mammograms with BIRADS label {0} : {1}".format(key, label_dict[key]))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title('Statistical analysis of the occurrence of BIRADS label')
    ax.bar(np.arange(len(occurrence)),
           np.asarray(occurrence),
           tick_label=[key for key in label_dict.keys()])

    plt.show()
    plt.close()

if __name__ == '__main__':
    # paths to csv files containing labels (and other information)
    csv_file_mass = \
        '/home/temp/moriz/data/all_mass_cases.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'

    #INbreast
    # train_paths, _, val_paths = inbreast_utils.load_single_set(inbreast_dir,
    #                                                            xls_file=xls_file,
    #                                                            train_size=0.7,
    #                                                            val_size=0.15,
    #                                                            random_state=42)
    #
    #
    dataset = LazyINbreastDataset(inbreast_dir,
                                  inbreast_utils.load_sample,
                                  xls_file = xls_file,
                                  #path_list= train_paths,
                                  type="all",
                                  label_type="birads",
                                  num_elements=10,
                                  detection_only = False)

    # DDSM
    # train_paths, test_paths, val_paths = ddsm_utils.load_single_set(ddsm_dir,
    #                                              csv_file=csv_file_mass,
    #                                              train_size=0.7,
    #                                              val_size=0.15,
    #                                              random_state=42)
    #
    # dataset = LazyDDSMDataset(ddsm_dir,
    #                           ddsm_utils.load_sample,
    #                           path_list=train_paths,
    #                           csv_file=csv_file_mass,
    #                           num_elements=None,
    #                           label_type="birads")


    main(dataset,
         image_save=False,
         image_save_dir = "/home/temp/moriz/")