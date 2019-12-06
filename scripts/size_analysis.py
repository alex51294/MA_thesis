import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import pickle

import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils

from detection.datasets import LazyDDSMDataset, CacheINbreastDataset, \
    LazyINbreastDataset

# maybe unnecessary, but did not work otherwise
backend = matplotlib.get_backend()
if backend != 'TkAgg':
    matplotlib.use('module://backend_interagg')

def main(dataset, image_save=False,
         image_save_dir="/home/temp/moriz", save_suffix=None):

    width_list = []
    height_list = []
    exceptions = []
    aspect_ratio = []
    for i in tqdm(range(len(dataset))):

        # sample = utils.segment_breast_via_hist(dataset[i])
        sample = dataset[i]

        channel, height, width = sample["data"].shape
        if height != sample["seg"].shape[1] or \
                width != sample["seg"].shape[2]:
            exceptions.append(i)
            continue
        else:
            width_list.append(width)
            height_list.append(height)
            aspect_ratio.append(height/width)

        # # evaluate lesion size
        # bbox = utils.bounding_box(dataset[i]["seg"])
        # for j in range(len(bbox)):
        #     width_list.append(bbox[j][2])
        #     height_list.append(bbox[j][3])

    print("Exceptions: {0}".format(exceptions))

    #data = np.transpose(np.asarray([width_list, height_list]))
    data = np.asarray(aspect_ratio)

    if image_save:
        with open(image_save_dir + save_suffix, "wb") as result_file:
            pickle.dump(data, result_file)

    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_title('Statistical analysis of the size of lesions')
    #ax.boxplot(data, labels=["width", "height"], whis=[2.5, 97.5])
    ax.boxplot(data, labels=["aspect ratio"], whis=[2.5, 97.5])
    if image_save:
        # create folder (if necessary)
        if not os.path.isdir(image_save_dir):
            os.makedirs(image_save_dir)

        if save_suffix is not None:
            save_path = image_save_dir + "/statistical_size_analysis_" + \
                        save_suffix + ".pdf"
        else:
            save_path = image_save_dir + "/statistical_size_analysis.pdf"

        plt.savefig(save_path, dpi='figure', format='pdf')
    plt.show()

    print("Minimum width: {0}".format(np.min(np.asarray(width_list))))
    print("Maximum width: {0}".format(np.max(np.asarray(width_list))))
    print("Minimum height: {0}".format(np.min(np.asarray(height_list))))
    print("Maximum height: {0}".format(np.max(np.asarray(height_list))))


if __name__ == '__main__':
    # paths to csv files containing labels (and other information)
    csv_mass_all = \
        '/home/temp/moriz/data/all_mass_cases.csv'
    csv_calc_all = \
        '/home/temp/moriz/data/all_calc_cases.csv'

    csv_calc_train = '/home/temp/moriz/data/calc_case_description_train_set.csv'
    csv_mass_train = '/home/temp/moriz/data/mass_case_description_train_set.csv'
    csv_calc_test = '/home/temp/moriz/data/calc_case_description_test_set.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'


    # INbreast
    # train_paths, _, _ = inbreast_utils.load_single_set(inbreast_dir,
    #                                                    xls_file=xls_file,
    #                                                    type="mass",
    #                                                    train_size=0.7,
    #                                                    val_size=0.15,
    #                                                    random_state=42)

    # # load INbreast data to test
    # dataset = LazyINbreastDataset(inbreast_dir,
    #                               inbreast_utils.load_sample,
    #                               #path_list = train_paths,
    #                               num_elements = None,
    #                               xls_file=xls_file,
    #                               type="mass",
    #                               segment = False,
    #                               img_shape=None,
    #                               shape_limit = None)

    # DDSM
    # train_paths, _, _ = ddsm_utils.load_single_set(ddsm_dir,
    #                                                csv_file=csv_mass_all,
    #                                                train_size=0.7,
    #                                                val_size=0.15,
    #                                                random_state=42)

    # load INbreast data to test
    dataset = LazyDDSMDataset(ddsm_dir,
                              ddsm_utils.load_sample,
                              #path_list=train_paths,
                              csv_file=csv_mass_train,
                              offset=None,
                              num_elements=10,
                              segment=False,
                              img_shape=None,
                              #shape_limit=[2600, 1300]
                              )

    main(dataset,
         image_save=False,
         image_save_dir = "/home/temp/moriz/dataset_analysis/ddsm/",
         save_suffix="aspect_ratio_unsegmented"
         )