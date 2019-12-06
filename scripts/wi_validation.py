import torch
import matplotlib.pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import os
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from detection.datasets import LazyDDSMDataset, CacheINbreastDataset, \
    LazyINbreastDataset, LazyDDSMDataset
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors
from scripts.paths import get_paths

def main(dataset,
         checkpoint_dir,
         start_epoch,
         end_epoch,
         step_size,
         results_save_dir="/home/temp/moriz/validation/pickled_results/",
         **settings):
    '''

    :param dataset: dataset to work with
    :param checkpoint_dir: path to checkpoint directory
    :param start_epoch: first model to validate
    :param end_epoch: last model to validate
    :param step_size: step size that determines in which intervals models
            shall be validated
    :param plot: flag
    :param offset: offset to validate only a part of the loaded data (
            usefull for debugging)
    '''

    # device
    device = 'cuda'
    #device = 'cpu'

    total_results_dict = {}

    # determine used set
    if "set" in settings and settings["set"] is not None:
        set = settings["set"]
    else:
        raise KeyError("Missing set description!")

    if "resnet" not in settings.keys():
        resnet = "RN50"
    else:
        resnet = settings["resnet"]

    # create directory and file name
    cp_dir_date = checkpoint_dir.split("/")[-3]
    if "fold" not in settings:
        results_save_dir = results_save_dir + \
                           str(cp_dir_date) + "/" + \
                           set + \
                           "/whole_image_level_" + \
                           str(start_epoch) + "_" + \
                           str(end_epoch) + "_" + \
                           str(step_size)
    else:
        results_save_dir = results_save_dir + \
                           str(cp_dir_date) + "/" + \
                           set + \
                           "/whole_image_level_" + \
                           str(start_epoch) + "_" + \
                           str(end_epoch) + "_" + \
                           str(step_size) + \
                           "/fold_" + str(settings["fold"])

    # create folder (if necessary)
    if not os.path.isdir(results_save_dir):
        os.makedirs(results_save_dir)

    # gather all important settings in one dict and save them (pickle them)
    settings_dict = {"level": "whole_image",
                     "checkpoint_dir": checkpoint_dir,
                     "start_epoch": start_epoch,
                     "end_epoch": end_epoch,
                     "step_size": step_size}
    settings_dict = {**settings_dict, **settings}

    with open(results_save_dir + "/settings", "wb") as settings_file:
        pickle.dump(settings_dict, settings_file)

    for epoch in tqdm(range(start_epoch, end_epoch + step_size, step_size)):
        checkpoint_path = checkpoint_dir + "/checkpoint_epoch_" + str(epoch) + ".pth"

        # load model
        if device == "cpu":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path)
        model = RetinaNet(**checkpoint['init_kwargs'], resnet=resnet).eval()
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.to(device)

        model_results_dict = {}

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                torch.cuda.empty_cache()


                # get image data
                test_data = dataset[i]["data"]
                if "crops" in dataset[i].keys():
                    crops = dataset[i]["crops"]
                    test_data = np.concatenate((test_data, crops), axis=0)

                gt_bbox = utils.bounding_box(dataset[i]["seg"])
                gt_label = dataset[i]["label"]

                # convert data to tensor to forward through the model
                torch.cuda.empty_cache()
                test_data = torch.Tensor(test_data).to(device)
                gt_bbox = torch.Tensor(gt_bbox).to(device)

                # predict anchors and labels for the crops using the loaded model
                anchor_preds, cls_preds = model(test_data.unsqueeze(0))

                # convert the predicted anchors to bboxes
                anchors = Anchors()
                boxes, labels, scores = anchors.generateBoxesFromAnchors(
                    anchor_preds[0],
                    cls_preds[0],
                    (test_data.shape[2], test_data.shape[1]),
                    cls_tresh=0.05)

                model_results_dict["image_%d" % i] = {"gt_list": gt_bbox,
                                                      "gt_label": gt_label,
                                                      "box_list": boxes,
                                                      "score_list": scores,
                                                      "labels_list": labels}
        # DATASET LEVEL
        total_results_dict[str(epoch)] = model_results_dict

    # MODELS LEVEL
    with open(results_save_dir + "/results", "wb") as result_file:
        torch.save(total_results_dict, result_file)

if __name__ == '__main__':
    # paths to csv files containing labels (and other information)
    csv_mass_all = '/home/temp/moriz/data/all_mass_cases.csv'
    csv_calc_all = '/home/temp/moriz/data/all_calc_cases.csv'

    csv_calc_train = '/home/temp/moriz/data/calc_case_description_train_set.csv'
    csv_calc_test = '/home/temp/moriz/data/calc_case_description_test_set.csv'

    csv_mass_train = '/home/temp/moriz/data/mass_case_description_train_set.csv'
    csv_mass_test = '/home/temp/moriz/data/mass_case_description_test_set.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'

    # path to image save directory
    image_save_dir = "/home/temp/moriz/validation/"

    # -----------------------------------------------------------------------
    # LOAD CHECKPOINT DIR CONTAINING EPOCHS FOR VALIDATION
    # -----------------------------------------------------------------------

    # 2600x1300, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-17_16-03-15")

    # 2600x1300, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-17_16-05-50")

    # 2600x1300, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-17_16-06-24")

    # 1800x900, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-17_21-41-08")

    # 1300x650, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-17_21-23-22")

    # 2600x1300, bs=1, lr=1e-4, segmented
    #checkpoint_dir = get_paths("19-04-17_21-36-36")

    # 1800x900, bs=1, lr=1e-4, segmented
    #checkpoint_dir = get_paths("19-04-17_21-35-44")

    # 1800x900, bs=2, lr=1e-4, segmented
    #checkpoint_dir = get_paths("19-04-17_21-34-16")

    # 1300x650, bs=2, lr=1e-4, segmented
    #checkpoint_dir = get_paths("19-04-17_21-28-44")

    # 1300x650, bs=1, lr=1e-4, segmented
    #checkpoint_dir = get_paths("19-04-19_21-57-44")

    #------------------------------------------------------------------------

    # RN152, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_20-26-01")

    # RN101, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_19-28-15")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_16-28-02")

    # RN34, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    # checkpoint_dir = get_paths("19-04-18_20-29-55")

    # RN18, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    # checkpoint_dir = get_paths("19-04-18_20-31-43")

    #------------------------------------------------------------------------

    # RN152, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-19_13-58-22")

    # RN101, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-19_14-00-07")

    # RN50, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_16-47-31")

    # RN34, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_20-35-16")

    # RN18, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_20-33-44")

    #-----------------------------------------------------------------------
    # RN50, IN, 1300x650, bs=1, lr=1e-5, unsegmented, sched.
    #checkpoint_dir = get_paths("19-04-19_21-48-21")

    # RN34, IN, 1300x650, bs=1, lr=1e-5, unsegmented, sched.
    #checkpoint_dir = get_paths("19-04-19_22-03-11")

    # RN18, IN, 1300x650, bs=1, lr=1e-5, unsegmented, sched.
    #checkpoint_dir = get_paths("19-04-19_22-04-44")

    #------------------------------------------------------------------------
    # RN18, IN, 2600x1300, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-19_16-12-30")

    # RN18, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-19_18-17-01")

    #-----------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-20_20-57-44")

    # RN50, IN, 1300x650, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-20_20-59-20")

    #-------------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=2, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-20_21-11-42")

    # RN50, IN, 1800x900, bs=2, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-20_19-28-56")

    # RN50, IN, 1800x900, bs=2, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-20_21-28-12")

    #-----------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=2, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-19_21-56-40")

    # RN50, IN, 1300x650, bs=2, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-20_21-01-16")

    # RN50, IN, 1300x650, bs=2, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-20_21-34-14")

    #-----------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=4, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-20_20-55-33")

    # RN50, IN, 1300x650, bs=4, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-20_19-30-50")

    # RN50, IN, 1300x650, bs=4, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-20_21-35-56")

    #-------------------------------------------------------------------------
    # RN50, IN, 2600x1300, bs=1, lr=1e-5, patho
    #checkpoint_dir = get_paths("19-04-21_21-00-57")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, birads
    #checkpoint_dir = get_paths("19-04-21_21-01-42")

    #-----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, seg
    #checkpoint_dir = get_paths("19-04-21_20-54-42")

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, seg
    #checkpoint_dir = get_paths("19-04-21_20-59-01")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, seg
    #checkpoint_dir = get_paths("19-04-21_21-06-40")

    # RN50, IN, 1300x650, bs=1, lr=1e-5, seg
    #checkpoint_dir = get_paths("19-04-21_21-07-38")

    #--------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size
    #checkpoint_dir = get_paths("19-04-22_15-32-16")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size, seg.
    #checkpoint_dir = get_paths("19-04-22_15-32-24")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex. size
    #checkpoint_dir = get_paths("19-04-22_21-41-07")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex. size, seg.
    #checkpoint_dir = get_paths("19-04-22_21-41-46")

    # RN50, IN, 1300x650, bs=1, lr=1e-5, flex. size
    #checkpoint_dir = get_paths("19-04-22_21-58-07")

    # RN50, IN, 1300x650, bs=1, lr=1e-5, flex. size, seg.
    #checkpoint_dir = get_paths("19-04-22_21-59-35")

    #----------------------------------------------------------------------

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 1080Ti
    #checkpoint_dir = get_paths("19-04-23_21-57-17")

    # RN50, IN, bs=1, lr=1e-5, flex. size, version 2, 1080Ti seg.
    #checkpoint_dir = get_paths("19-04-23_22-00-54")

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 980Ti
    #checkpoint_dir = get_paths("19-04-24_18-48-20")

    # RN50, IN, bs=1, lr=1e-5, flex. size, version 2, 980Ti seg.
    #checkpoint_dir = get_paths("19-04-24_18-49-34")

    # RN50, IN, bs=1, lr=1e-5, flex. size, version 2, 980Ti seg. test
    #checkpoint_dir = get_paths("19-04-28_13-52-53")

    # RN50, IN, bs=1, lr=1e-5, flex. size, version 2, 1080Ti seg. test
    #checkpoint_dir = get_paths("19-04-28_13-54-55")

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, better aug.
    #checkpoint_dir = get_paths("19-04-28_21-24-09")

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, new seg. test
    #checkpoint_dir = get_paths("19-04-30_20-28-21")

    #-----------------------------------------------------------------------

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, calc.
    #checkpoint_dir = get_paths("19-04-29_09-47-38")

    # RN50, IN, 2600x1300, lr=1e-4, calc.
    #checkpoint_dir = get_paths("19-05-01_23-19-03")

    # RN50, IN, 2600x1300, lr=1e-5, calc.
    #checkpoint_dir = get_paths("19-05-01_23-18-12")

    # RN50, IN, 2600x1300, lr=1e-6, calc.
    #checkpoint_dir = get_paths("19-05-01_23-20-26")

    #------------------------------------------------------------------------

    # RN18, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-30_19-47-44")

    # RN18, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-30_19-39-47")

    # RN34, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-30_19-53-14")

    # RN34, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    # checkpoint_dir = get_paths("19-04-30_19-38-27")

    # RN101, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-30_19-45-58")

    # RN101, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-30_19-35-09")

    # RN152, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-05-01_10-38-36")

    # RN152, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-30_19-32-54")

    #-----------------------------------------------------------------------
    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, seg. test
    #checkpoint_dir = get_paths("19-05-01_22-26-23")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex, seg. test
    #checkpoint_dir = get_paths("19-05-01_22-27-06")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, fix, seg. test
    #checkpoint_dir = get_paths("19-05-01_22-29-19")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, fix, seg. test
    #checkpoint_dir = get_paths("19-05-01_22-28-33")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, better aug.
    #checkpoint_dir = get_paths("19-05-01_22-34-30")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, seg. test, better aug.
    #checkpoint_dir = get_paths("19-05-01_22-32-48")

    #----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, fix, better aug.
    #checkpoint_dir = get_paths("19-05-03_14-44-34")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, fix, seg., better aug.
    #checkpoint_dir = get_paths("19-05-03_14-45-20")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, fix, better aug.
    #checkpoint_dir = get_paths("19-05-03_15-16-15")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex, better aug.
    #checkpoint_dir = get_paths("19-05-03_15-11-55")

    # ----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, patho.
    #checkpoint_dir = get_paths("19-05-03_23-55-21")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, birads
    #checkpoint_dir = get_paths("19-05-03_23-58-08")

    #---------------------------------------------------------------------

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, flex, (baseline)
    #checkpoint_dir = get_paths("19-05-05_15-25-37")

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, flex, CL
    # checkpoint_dir = get_paths("19-05-01_11-25-14")

    # RN18, IN, 2600x1300, bs=1, lr=1e-6, flex, CL
    # checkpoint_dir = get_paths("19-05-01_11-25-55")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, CL
    #checkpoint_dir = get_paths("19-05-01_11-22-10")

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, flex, CL
    #checkpoint_dir = get_paths("19-05-01_11-23-58")

    #-------------------------------------------------------------------------

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, (baseline)
    #checkpoint_dir = get_paths("19-05-05_16-55-51")

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, CL
    #checkpoint_dir = get_paths("19-05-05_16-43-00")

    # RN18, IN, 2600x1300, bs=1, lr=1e-6, CL
    #checkpoint_dir = get_paths("19-05-05_16-47-51")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, CL
    #checkpoint_dir = get_paths("19-05-05_16-51-09")

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, CL
    checkpoint_dir = get_paths("19-05-05_16-52-08")

    # -------------------------------------------------------------------------

    # INbreast RN50, 2600x1300 IN, bs=1, lr=1e-4,
    #checkpoint_dir = get_paths("19-04-24_18-55-11")

    # INbreast RN50, 2600x1300 IN, bs=1, lr=1e-5,
    #checkpoint_dir = get_paths("19-04-23_10-35-57")

    # INbreast RN50, 2600x1300 IN, bs=1, lr=1e-6,
    #checkpoint_dir = get_paths("19-04-24_18-54-06")

    # INbreast RN50, 2600x1300, bs=1, lr=1e-5, seg.
    #checkpoint_dir = get_paths("19-04-23_10-37-23")

    #--------------------------------------------------------------------------

    # flags and options
    settings = {"segment": False,
                "img_shape": [2600, 1300],
                #"shape_limit": [2600, 1300],
                #"shape_limit": "1080Ti",
                "random_seed": 42,
                "norm": True,
                "type": "mass",
                #"crop_number": 9,
                "num_elements": None,
                "detection_only": True,
                "label_type": None,
                "resnet": "RN50",
                "set": "val"}


    # single set
    # _, _, val_paths = \
    #     inbreast_utils.load_single_set(inbreast_dir,
    #                                    xls_file=xls_file,
    #                                    train_size=0.7,
    #                                    val_size=0.15,
    #                                    type=settings["type"],
    #                                    random_state=settings["random_seed"])
    #
    # # load INbreast data
    # dataset = LazyINbreastDataset(inbreast_dir,
    #                               inbreast_utils.load_sample,
    #                               path_list=val_paths,
    #                               xls_file=xls_file,
    #                               **settings)

    # DDSM
    _, val_paths, _ = ddsm_utils.load_single_set(ddsm_dir,
                                                 csv_file=csv_mass_train,
                                                 #csv_file=csv_calc_train,
                                                 train_size=0.9,
                                                 val_size=None,
                                                 random_state=settings["random_seed"])

    dataset = LazyDDSMDataset(ddsm_dir,
                              ddsm_utils.load_sample,
                              #ddsm_utils.load_sample_with_crops,
                              path_list=val_paths,
                              csv_file=csv_mass_train,
                              #csv_file=csv_calc_train,
                              **settings)

    main(dataset,
         checkpoint_dir=checkpoint_dir,
         start_epoch=2,
         end_epoch=50,
         step_size=2,
         results_save_dir="/home/temp/moriz/validation/results/",
         **settings)

    # kfold
    # paths = inbreast_utils.get_paths(inbreast_dir,
    #                                  xls_file=xls_file,
    #                                  type=settings["type"])
    #
    # train_splits, _ = utils.kfold_patientwise(paths,
    #                                           dataset_type="INbreast",
    #                                           num_splits=5,
    #                                           shuffle=True,
    #                                           random_state=settings["random_seed"])
    #
    # for i in tqdm(range(5)):
    #     # flags and options
    #     settings["fold"] = i
    #
    #     _, val_paths, _ = \
    #         utils.split_paths_patientwise(train_splits[i],
    #                                       dataset_type="INbreast",
    #                                       train_size=0.9)
    #
    #     # load INbreast data
    #     dataset = LazyINbreastDataset(inbreast_dir,
    #                                   inbreast_utils.load_sample,
    #                                   path_list=val_paths,
    #                                   xls_file=xls_file,
    #                                   **settings)
    #
    #     main(dataset,
    #          checkpoint_dir=checkpoint_dir + "run_0%d" % i,
    #          start_epoch=2,
    #          end_epoch=50,
    #          step_size=2,
    #          results_save_dir="/home/temp/moriz/validation/results/",
    #          **settings)
