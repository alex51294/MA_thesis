import torch
from tqdm import tqdm
import numpy as np
import os
import pickle
import time

import scripts.eval_utils as eval_utils
import scripts.plot_utils as plot_utils

def main(results_dir,
         image_save=False,
         model_epochs=None,
         image_save_dir = "/home/temp/moriz/validation",
         plot_froc=False,
         plot_f1_score=False,
         plot_ap=False,
         merging_method="NMS",
         score_thr=0.0,
         format="svg"):

    #results_dir = "/home/temp/moriz/validation/pickled_results/" + results_dir

    # load settings
    with open(results_dir + "/settings", "rb") as settings_file:
        settings_dict = pickle.load(settings_file)

    # display settings
    for key in settings_dict.keys():
        print("{0}: {1}".format(key, settings_dict[key]))

    # set start, end and step size of the desired evaluation
    start_epoch = max(settings_dict["start_epoch"], model_epochs[0])
    end_epoch = min(settings_dict["end_epoch"], model_epochs[1])
    step_size = max(settings_dict["step_size"], model_epochs[2])
    eval_keys = np.arange(start_epoch, end_epoch + step_size, step_size)
    confidence_values = np.arange(0.05, 1., 0.05)
    level = settings_dict["level"]

    start_time = time.time()
    # load the actual results
    with open(results_dir + "/results", "rb") as file:
        #result_dict = pickle.load(file)
        #result_dict = torch.load(file)
        result_dict = torch.load(file, map_location="cpu")
    print(time.time() - start_time)

    f1_scores_list = []
    froc_tpr_list = []
    froc_fppi_list = []
    ap_list = []

    # iterate over the keys (i.e. epochs) in the result file
    for key in tqdm(result_dict.keys()):

        # choose only the ones that are targeted for the evaluation
        if int(key) not in eval_keys:
            continue

        # each saved epoch is regarded as model
        model_dict = result_dict[key]

        tp_list = []
        fp_list = []
        fn_list = []

        box_labels_list = []
        scores_list = []
        iou_hits_list = []

        num_gt = 0

        # each model dict has an image dict, containing a  gt_list,
        # a box_list, a score_list and a dict for additional merging
        # information (required for WBC)
        for image_key in model_dict.keys():
            gt_list = model_dict[image_key]["gt_list"]
            box_list = model_dict[image_key]["box_list"]
            score_list = model_dict[image_key]["score_list"]

            if "labels_list" in model_dict[image_key]:
                labels_list = model_dict[image_key]["labels_list"]
            else:
                labels_list = None

            num_gt += len(gt_list)

            # if detections present, compare them to the ground truth
            if box_list is not None and len(box_list) > 0:
                # merge overlapping bounding boxes using NMS
                if level == "image" and merging_method == "NMS":
                    box_list, score_list, labels_list = \
                        eval_utils.nms(box_list,
                                       score_list,
                                       labels_list,
                                       0.2)

                elif level == "image" and merging_method == "WBC":
                    ccf = model_dict[image_key]["merging_utils"]["ccf"]
                    hf = model_dict[image_key]["merging_utils"]["hf"]

                    box_list, score_list = eval_utils.my_merging(box_list,
                                                                 score_list,
                                                                 ccf,
                                                                 hf,
                                                                 thr=0.2)

                box_labels, scores, iou_hits = \
                    eval_utils.calc_detection_hits(gt_list,
                                                   box_list,
                                                   score_list,
                                                   score_thr=score_thr)
                box_labels_list.append(box_labels)
                scores_list.append(scores)
                iou_hits_list.append(iou_hits)

                # calculate TP, FP and FN with regard to the ground truth
                tp, fp, fn = \
                    eval_utils.calc_tp_fn_fp(gt_list,
                                             box_list,
                                             score_list,
                                             confidence_values=confidence_values)

                # add the image information to the list
                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
            else:
                tp_list.append([torch.tensor(0) for tp
                                in range(len(confidence_values))])
                fp_list.append([torch.tensor(0) for fp
                                in range(len(confidence_values))])
                fn_list.append([torch.tensor(len(gt_list)) for fn
                                in range(len(confidence_values))])

        if plot_f1_score:
            f1_scores = eval_utils.calc_f_beta(tp_list, fp_list, fn_list, beta=1)
            #f1_scores = eval_utils.calc_f1(tp_list, fp_list, fn_list)
            f1_scores_list.append(f1_scores)

        if plot_froc:
            froc_tpr, froc_fppi = eval_utils.calc_froc(tp_list, fp_list,
                                                       fn_list)
            froc_tpr_list.append(froc_tpr)
            froc_fppi_list.append(froc_fppi)

        if plot_ap:
            #print(num_gt)
            ap, _ = eval_utils.calc_ap_MDT(box_labels_list, scores_list, num_gt)
            ap_list.append(ap)

    if image_save:
        if "fold" in settings_dict:
            image_save_dir += "/fold_" + str(settings_dict["fold"]) + "/" + \
                              settings_dict["set"] + "_" + \
                              level + "_" + str(start_epoch) + \
                              "_" + str(end_epoch) + "_" + str(step_size)

        else:
            image_save_dir += settings_dict["set"] + "_" + \
                              level + "_" + str(start_epoch) + \
                              "_" + str(end_epoch) + "_" + str(step_size)

        if not os.path.isdir(image_save_dir):
                os.makedirs(image_save_dir)

    if plot_f1_score:
        plot_utils.plot_f1(f1_scores_list, confidence_values,
                           models=[start_epoch, end_epoch, step_size],
                           image_save=image_save,
                           image_save_dir=image_save_dir,
                           plot_average=True,
                           plot_maximum=True,
                           plot_all=False,
                           format=format)

    if plot_froc:
        plot_utils.plot_frocs(froc_tpr_list, froc_fppi_list,
                              models=[start_epoch, end_epoch, step_size],
                              image_save=image_save,
                              image_save_dir=image_save_dir,
                              left_range=1e-2,
                              format=format)

    if plot_ap:
        plot_utils.plot_ap(ap_list, models=[start_epoch, end_epoch, step_size],
                           image_save=image_save,
                           image_save_dir=image_save_dir,
                           save_suffix=merging_method,
                           format=format)


if __name__ == '__main__':

    #--------------------------------------------------------------------------
    # PATHS TO DIRS CONTAINING VALIDATION RESULTS
    #-----------------------------------------------------------------------

    # 2600x1300, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_16-03-15/val/whole_image_level_5_50_5"

    # 2600x1300, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_16-05-50/val/whole_image_level_5_50_5"

    # 2600x1300, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_16-06-24/val/whole_image_level_5_50_5"

    # 1800x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_21-41-08/val/whole_image_level_5_50_5"

    # 1300x650, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_21-23-22/val/whole_image_level_5_50_5"

    # 2600x1300, bs=1, lr=1e-4, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_21-36-36/val/whole_image_level_5_50_5"

    # 1800x900, bs=2, lr=1e-4, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_21-34-16/val/whole_image_level_5_50_5"

    # 1800x900, bs=1, lr=1e-4, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_21-35-44/val/whole_image_level_5_50_5"

    # 1300x650, bs=1, lr=1e-4, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_21-57-44/val/whole_image_level_4_48_4"

    # 1300x650, bs=2, lr=1e-4, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-17_21-28-44/val/whole_image_level_5_50_5"

    #------------------------------------------------------------------------

    # RN152, IN, 1800x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_20-26-01/val/whole_image_level_5_50_5"

    # RN101, IN, 1800x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_19-28-15/val/whole_image_level_5_50_5"

    # RN50, IN, 1800x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_16-28-02/val/whole_image_level_5_50_5"

    # RN34, IN, 1800x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_20-29-55/val/whole_image_level_5_50_5"

    # RN18, IN, 1800x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_20-31-43/val/whole_image_level_5_50_5"

    #------------------------------------------------------------------------

    # RN18, IN, 1800x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-47-44/val/whole_image_level_4_48_4"

    # RN18, IN, 1800x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-39-47/val/whole_image_level_4_48_4"

    # RN34, IN, 1800x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-53-14/val/whole_image_level_4_48_4"

    # RN34, IN, 1800x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-38-27/val/whole_image_level_4_48_4"

    # RN101, IN, 1800x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-45-58/val/whole_image_level_4_48_4"

    # RN101, IN, 1800x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-35-09/val/whole_image_level_4_40_4"

    # RN152, IN, 1800x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_10-38-36/val/whole_image_level_4_48_4"

    # RN152, IN, 1800x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_19-32-54/val/whole_image_level_4_36_4"

    # ------------------------------------------------------------------------

    # RN152, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_13-58-22/val/whole_image_level_5_50_5"

    # RN101, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_14-00-07/val/whole_image_level_5_50_5"

    # RN50, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_16-47-31/val/whole_image_level_5_50_5"

    # RN34, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_20-35-16/val/whole_image_level_5_50_5"

    # RN18, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-18_20-33-44/val/whole_image_level_5_50_5"

    #------------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_21-48-21/val/whole_image_level_2_50_2"

    # RN34, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_22-03-11/val/whole_image_level_2_50_2"

    # RN18, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_22-04-44/val/whole_image_level_2_50_2"

    #-----------------------------------------------------------------------

    # RN18, IN, 1800x900, bs=1, lr=1e-5, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_16-12-30/val/whole_image_level_5_20_5"

    # RN18, IN, 1300x650, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_18-17-01/val/whole_image_level_2_12_2"

    #----------------------------------------------------------------------

    # RN50, IN,600x600, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_12-59-19/val/image_level_5_25_5"

    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_12-59-19/val/image_level_25_50_5"

    # RN50, IN, 900x900, bs=1, lr=1e-5, broken
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_13-01-54/val/image_level_5_15_5"

    # RN50, IN, 900x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-24_12-28-33/val/image_level_4_24_4"

    # RN50, IN, 900x900, bs=1, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_21-16-26/val/image_level_2_24_2"

    # RN50, IN, 900x900, bs=2, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_20-42-38/val/image_level_4_24_4"

    # RN50, IN, 900x900, bs=4, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_20-41-55/val/image_level_2_24_2"

    # RN50, IN, 1200x1200, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_13-02-56/val/image_level_5_25_5"

    # RN50, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-55-18/val/image_level_4_20_4"

    # RN50, IN, 900x900, bs=1, lr=1e-4, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-54-47/val/image_level_2_20_2"

    # RN50, IN, 900x900, bs=1, lr=1e-6, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-53-52/val/image_level_2_20_2"

    # RN50, IN, 900x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-52-34/val/image_level_4_20_4"

    # RN18, IN, 900x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_20-44-17/val/image_level_2_24_2"

    # RN34, IN, 900x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-24_12-29-08/val/image_level_4_24_4"

    # -----------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_20-57-44/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_20-59-20/val/whole_image_level_2_50_2"

    #----------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=2, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_21-11-42/val/whole_image_level_2_50_2"

    # RN50, IN, 1800x900, bs=2, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_19-28-56/val/whole_image_level_2_50_2"

    # RN50, IN, 1800x900, bs=2, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_21-28-12/val/whole_image_level_2_50_2"

    #----------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=2, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-19_21-56-40/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=2, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_21-01-16/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=2, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_21-34-14/val/whole_image_level_2_50_2"

    #----------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=4, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_20-55-33/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=4, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_19-30-50/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=4, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-20_21-35-56/val/whole_image_level_2_50_2"

    #----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_20-54-42/val/whole_image_level_2_24_2"

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_20-59-01/val/whole_image_level_2_24_2"

    # RN50, IN, 1800x900, bs=1, lr=1e-5, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_21-06-40/val/whole_image_level_2_24_2"

    # RN50, IN, 1300x650, bs=1, lr=1e-5, seg
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_21-07-38/val/whole_image_level_2_24_2"

    #----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, patho
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_21-00-57/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, birads
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-21_21-01-42/val/whole_image_level_2_50_2"

    #----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_15-32-16/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size, seg.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_15-32-24/val/whole_image_level_2_50_2"

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex. size
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-41-07/val/whole_image_level_2_50_2"

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex. size, seg.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-41-46/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=1, lr=1e-5, flex. size
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-58-07/val/whole_image_level_2_50_2"

    # RN50, IN, 1300x650, bs=1, lr=1e-5, flex. size, seg.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-22_21-59-35/val/whole_image_level_2_26_2"

    # RN50, IN, bs=1, lr=1e-5, flex. size, 1080Ti
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-23_21-57-17/val/whole_image_level_2_30_2"

    # RN50, IN, bs=1, lr=1e-5, flex. size, 1080Ti, seg.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-23_22-00-54/val/whole_image_level_2_30_2"

    # RN50, IN, bs=1, lr=1e-5, flex. size, 980Ti
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-24_18-48-20/val/whole_image_level_2_50_2"

    # RN50, IN, bs=1, lr=1e-5, flex. size, 980Ti, seg.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-24_18-49-34/val/whole_image_level_2_50_2"

    # ----------------------------------------------------------------------

    # RN18, IN, 900x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_18-31-41/val/image_level_4_48_4"

    # RN34, IN, 900x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-26-57/val/image_level_4_32_4"

    # RN50, IN, 900x900, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-27-35/val/image_level_4_32_4"

    #-----------------------------------------------------------------------

    # RN18, IN, 900x900, bs=2, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_18-36-27/val/image_level_4_48_4"

    # RN18, IN, 900x900, bs=4, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_19-30-10/val/image_level_4_48_4"

    # RN18, BN, 900x900, bs=4, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_19-02-14/val/image_level_4_48_4"

    #-----------------------------------------------------------------------

    # RN18, IN, 600x600, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_18-30-29/val/image_level_4_24_4"

    # RN18, IN, 600x600, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_18-30-29/val/image_level_24_48_4"

    #-----------------------------------------------------------------

    # RN18, IN, 1200x1200, bs=1, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-26_18-34-13/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=2, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-58-38/val/image_level_4_36_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-56-50/val/image_level_4_28_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-29_22-34-07/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=8, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-50-45/val/image_level_4_48_4"

    # RN18, BN, 1200x1200, bs=8, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_23-11-34/val/image_level_4_48_4"

    #---------------------------------------------------------------------

    # RN18, IN, 1200x1200, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_17-01-16/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=2, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-59-57/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-55-51/val/image_level_4_28_4"

    # RN18, IN, 1200x1200, bs=8, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_16-52-23/val/image_level_2_10_2"

    # RN18, IN, 1200x1200, bs=8, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-29_22-31-29/val/image_level_4_48_4"

    # RN18, BN, 1200x1200, bs=8, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_23-15-01/val/image_level_4_48_4"

    # ------------------------------------------------------------------------

    # RN18, IN, 900x900, bs=1, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-04_17-11-20/val/image_level_4_24_4"

    # RN18, IN, 900x900, bs=1, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-04_17-11-20/val/image_level_24_48_4"

    # RN18, IN, 1200x1200, bs=1, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_23-00-22/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=1, lr=1e-6, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-27_23-01-58/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=1, lr=1e-6, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-04_18-10-38/val/image_level_4_32_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-29_22-27-26/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-04_17-13-12/val/image_level_4_20_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-04_17-13-12/val/image_level_20_36_4"

    #---------------------------------------------------------------------

    # RN50, IN, WI full flex, bs=1, lr=1e-5, test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-28_13-52-53/val/whole_image_level_2_20_2"

    # RN50, IN, WI full flex, bs=1, lr=1e-5, test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-28_13-54-55/val/whole_image_level_2_20_2"

    # RN50, IN, WI 2600x1300 half flex, lr=1e-5, better aug
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-28_21-24-09/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, new seg. test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-30_20-28-21/val/whole_image_level_2_50_2"

    # ---------------------------------------------------------------------

    # RN50, IN, WI 2600x1300 half flex, lr=1e-5, calc.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-04-29_09-47-38/val/whole_image_level_2_50_2"

    # RN50, IN, WI 2600x1300 fix, lr=1e-4, calc.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_23-19-03/val/whole_image_level_2_50_2"

    # RN50, IN, WI 2600x1300 fix, lr=1e-5, calc.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_23-18-12/val/whole_image_level_2_50_2"

    # RN50, IN, WI 2600x1300 fix, lr=1e-6, calc.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_23-20-26/val/whole_image_level_2_50_2"

    #----------------------------------------------------------------------

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, new seg. test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_22-26-23/val/whole_image_level_2_50_2"

    # RN50, IN, 1800x900, one side flex, lr=1e-5, new seg. test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_22-27-06/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, fix, lr=1e-5, new seg. test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_22-29-19/val/whole_image_level_2_50_2"

    # RN50, IN, 1800x900, fix, lr=1e-5, new seg. test
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_22-28-33/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_22-34-30/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, seg. test, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_22-32-48/val/whole_image_level_2_50_2"

    # # RN50, IN, 2600x1300, fix, lr=1e-5, better aug.
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_14-44-34/val/whole_image_level_2_50_2"

    #-------------------------------------------------------------------

    # RN50, IN, bs=1, lr=1e-5, flex, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_11-22-10/val/whole_image_level_2_50_2"

    # RN50, IN, bs=1, lr=1e-6, flex, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_11-23-58/val/whole_image_level_2_50_2"

    # RN18, IN, bs=1, lr=1e-5, flex, comparison for CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-05_15-25-37/val/whole_image_level_2_50_2"

    # RN18, IN, bs=1, lr=1e-5, flex, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_11-25-14/val/whole_image_level_2_50_2"

    # RN18, IN, bs=1, lr=1e-6, flex, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-01_11-25-55/val/whole_image_level_2_50_2"

    #--------------------------------------------------------------------------

    # RN50, IN, bs=1, lr=1e-5, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-05_16-51-09/val/whole_image_level_2_50_2"

    # RN50, IN, bs=1, lr=1e-6, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-05_16-52-08/val/whole_image_level_2_50_2"

    # RN18, IN, bs=1, lr=1e-5, comparison for CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-05_16-55-51/val/whole_image_level_2_50_2"

    # RN18, IN, bs=1, lr=1e-5, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-05_16-43-00/val/whole_image_level_2_50_2"

    # RN18, IN, bs=1, lr=1e-6, CL
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-05_16-47-51/val/whole_image_level_2_50_2"

    # -------------------------------------------------------------------------

    # RN18, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-02_12-13-43/val/image_level_4_48_4"

    # RN18, IN, 900x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_23-52-19/val/image_level_4_24_4"

    # RN18, IN, 900x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_23-52-19/val/image_level_24_48_4"

    # RN34, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-02_12-09-44/val/image_level_4_24_4"

    # RN34, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-02_12-09-44/val/image_level_24_48_4"

    # RN34, IN, 900x900, bs=1, lr=1e-6
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-02_12-11-00/val/image_level_4_48_4"

    # RN50, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_00-14-58/val/image_level_4_48_4"

    # RN50, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_00-17-30/val/image_level_4_24_4"

    # RN50, IN, 900x900, bs=1, lr=1e-4
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_00-17-30/val/image_level_24_48_4"

    # RN18, IN, 900x900, bs=8, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-02_19-11-02/val/image_level_4_48_4"

    # RN18, BN, 900x900, bs=8, lr=1e-5
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-02_19-27-14/val/image_level_4_48_4"

    #------------------------------------------------------------------------

    # RN18, IN, 600x600, bs=1, lr=1e-5, calc
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_00-10-53/val/image_level_4_48_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, calc
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-07_15-04-32/val/image_level_4_24_4"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, calc
    result_dir = "/home/temp/moriz/validation/results/" \
                 "19-05-07_15-04-32/val/image_level_24_48_4"

    # RN50, IN, 2600x1300, flex, lr=1e-5, patho
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_23-55-21/val/whole_image_level_2_50_2"

    # RN50, IN, 2600x1300, flex, lr=1e-5, birads
    # result_dir = "/home/temp/moriz/validation/results/" \
    #              "19-05-03_23-58-08/val/whole_image_level_2_30_2"


    # path to image save directory
    # image_save_dir = "/home/temp/moriz/validation/plots/new/" + \
    #                  result_dir.split("/")[-3] + "/"

    # image_save_dir = "/home/temp/moriz/validation/new_plots/" + \
    #                  result_dir.split("/")[-3] + "/"

    # image_save_dir = "/home/temp/moriz/validation/tmp/" + \
    #                  result_dir.split("/")[-3] + "/"
    #
    image_save_dir = "/home/temp/moriz/validation/final/val_plots/" + \
                     result_dir.split("/")[-3] + "/"

    main(result_dir,
         plot_f1_score=True,
         plot_froc=True,
         plot_ap=True,
         model_epochs=[4, 50, 4],
         image_save=False,
         image_save_dir=image_save_dir,
         merging_method="NMS",
         score_thr=0.0,
         format="pdf",
         )

    # ------------------------------------------------------------------------
    # INbreast k-fold

    # for i in range(5):
    #     # 1800x900
    #     # result_dir = "/home/temp/moriz/validation/results/" \
    #     #              "19-04-06_15-33-06/val/whole_image_level_5_200_5/fold_%d" % i
    #
    #     # 1300x650
    #     # result_dir = "/home/temp/moriz/validation/results/" \
    #     #              "19-04-06_15-22-18/val/whole_image_level_5_200_5/fold_%d" % i
    #
    #     # 600x600
    #     # result_dir = "/home/temp/moriz/validation/test/" \
    #     #              "19-04-07_14-31-24/val/image_level_5_250_5/fold_%d" % i
    #
    #     # # 900x900
    #     # result_dir = "/home/temp/moriz/validation/test/" \
    #     #              "19-04-07_14-48-22/val/image_level_5_250_5/fold_%d" % i
    #
    #     # # 2600x1300, 1e-4
    #     # result_dir = "/home/temp/moriz/validation/results/" \
    #     #              "19-04-24_18-55-11/val/whole_image_level_2_50_2/fold_%d" % i
    #
    #     # 2600x1300, 1e-5
    #     # result_dir = "/home/temp/moriz/validation/results/" \
    #     #              "19-04-23_10-35-57/val/whole_image_level_2_50_2/fold_%d" % i
    #
    #     # 2600x1300, 1e-6
    #     result_dir = "/home/temp/moriz/validation/results/" \
    #                  "19-04-24_18-54-06/val/whole_image_level_2_50_2/fold_%d" % i
    #
    #     # 1200x1200
    #     # result_dir = "/home/temp/moriz/validation/final/test_results/" \
    #     #              "19-04-29_09-57-24/val/image_level_2_50_2/fold_%d" % i
    #
    #     # 900x900
    #     # result_dir = "/home/temp/moriz/validation/final/test_results/" \
    #     #              "19-04-29_09-58-41/val/image_level_2_50_2/fold_%d" % i
    #
    #     # 600x600
    #     # result_dir = "/home/temp/moriz/validation/final/results/" \
    #     #              "19-05-04_18-14-50/val/image_level_2_50_2/fold_%d" % i
    #
    #     # path to image save directory
    #     image_save_dir = "/home/temp/moriz/validation/final/val_plots/" + \
    #                      result_dir.split("/")[-4] + "/"
    #
    #     main(result_dir,
    #          plot_f1_score=True,
    #          plot_froc=True,
    #          plot_ap=True,
    #          model_epochs=[2, 50, 2],
    #          image_save=True,
    #          image_save_dir=image_save_dir,
    #          merging_method="NMS",
    #          score_thr=0.0,
    #          format="pdf",
    #          )
