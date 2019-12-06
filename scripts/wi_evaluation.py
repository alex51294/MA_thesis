import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os

from tqdm import tqdm
import numpy as np
import pickle

from detection.datasets import LazyINbreastDataset, LazyDDSMDataset
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors
from scripts.paths import get_paths

import scripts.eval_utils as eval_utils
import scripts.plot_utils as plot_utils

def main(dataset,
         model_path,
         plot_image=False,
         image_save=False,
         image_save_dir = None,
         image_save_subdir = None,
         image_save_suffix = None,
         calc_ap=False,
         calc_froc=False,
         calc_f1=False,
         eval_class=False,
         NMS_thr=0.5,
         **settings):

    # device
    device = 'cuda'

    if "resnet" not in settings.keys():
        resnet = "RN50"
    else:
        resnet = settings["resnet"]

    # load model
    checkpoint = torch.load(model_path)
    model = RetinaNet(**checkpoint['init_kwargs'], resnet=resnet)
    model.eval()
    model.load_state_dict(checkpoint['state_dict']["model"])
    model.to(device)

    if image_save:
        path_parts = model_path.split("/")
        if image_save_dir is None:
            image_save_dir = "/home/temp/moriz/validation/test_results/" + \
                             path_parts[-4] + "/" + path_parts[-2] + "/" + \
                             path_parts[-1].split(".")[0] + "/"
        else:
            image_save_dir += path_parts[-4] + "/" + path_parts[-2] + "/" + \
                             path_parts[-1].split(".")[0] + "/"

        if image_save_subdir is not None:
            image_save_dir += "/" + image_save_subdir + "/"

        if not os.path.isdir(image_save_dir):
            os.makedirs(image_save_dir)

    tp_list = []
    fp_list = []
    fn_list = []

    image_class_list = []
    ribli_class_list = []

    box_label_list = []
    box_score_list = []

    confidence_values = np.arange(0.05, 1, 0.05)
    num_gt = 0

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
            num_gt += len(gt_bbox)

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
                cls_tresh=0.05,
                nms_thresh=NMS_thr)

            #boxes, scores = eval_utils.wbc(boxes, scores, thr=0.2)
            #boxes, scores, labels = eval_utils.nms(boxes, scores, labels, thr=0.2)
            # boxes, scores, labels = eval_utils.rm_overlapping_boxes(boxes,
            #                                                         scores,
            #                                                         labels,
            #                                                         order="xywh")

            if boxes is None:
                boxes = []
                tp_list.append([torch.tensor(0, device=device) for tp
                                in range(len(confidence_values))])
                fp_list.append([torch.tensor(0, device=device) for fp
                                in range(len(confidence_values))])
                fn_list.append([torch.tensor(len(gt_bbox), device=device) for fn
                                in range(len(confidence_values))])

            else:
                # calculate the required rates for the FROC metric
                tp_crop, fp_crop, fn_crop = \
                    eval_utils.calc_tp_fn_fp(gt_bbox,
                                             boxes,
                                             scores,
                                             confidence_values=confidence_values)
                tp_list.append(tp_crop)
                fp_list.append(fp_crop)
                fn_list.append(fn_crop)

                # determine the overlap of detected bbox with the ground truth
                box_label, box_score, _ = \
                    eval_utils.calc_detection_hits(gt_bbox,
                                                   boxes,
                                                   scores,
                                                   score_thr=0.0)
                box_label_list.append(box_label)
                box_score_list.append(box_score)

                image_class_list.append([gt_label,
                                         np.float32(labels[torch.argmax(scores)].cpu()),
                                         np.float32(torch.max(scores).cpu())])


            # plot the image with the according bboxes
            if plot_image:
                test_data = test_data.to("cpu")
                gt_bbox = gt_bbox.to("cpu")

                # plot image
                c, h, w = test_data.shape
                #figsize = 0.75 * (w / 100), 0.75 * (h / 100)
                figsize = 0.5 * (w / 100), 0.5 * (h / 100)
                #figsize = 0.25 * (w / 100), 0.25 * (h / 100)
                fig, ax = plt.subplots(1, figsize=figsize)

                ax.imshow(test_data[0, :, :], cmap='Greys_r')

                # show bboxes as saved in data (in red with center)
                for l in range(len(gt_bbox)):
                    pos = tuple(gt_bbox[l][0:2])
                    plt.plot(pos[0], pos[1], 'r.')
                    width = gt_bbox[l][2]
                    height = gt_bbox[l][3]
                    pos = (pos[0] - np.floor(width / 2),
                           pos[1] - np.floor(height / 2))

                    # Create a Rectangle patch
                    rect = patches.Rectangle(pos, width, height, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.add_patch(rect)
                    ax.annotate("{:d}".format(np.int32(gt_label)),
                                pos,
                                fontsize=6,
                                color="r",
                                xytext=(pos[0] - 10, pos[1] - 10))

                # show the predicted bboxes (in blue)
                print("Number of detected bboxes: {0}".format(len(boxes)))
                # keep = scores > 0.15
                # boxes = boxes[keep]
                # scores = scores[keep]
                for j in range(len(boxes)):
                    width = boxes[j][2]
                    height = boxes[j][3]
                    pos = (boxes[j][0] - torch.floor(width / 2),
                           boxes[j][1] - torch.floor(height / 2))

                    # Create a Rectangle patch
                    rect = patches.Rectangle(pos, width, height, linewidth=1,
                                             edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    ax.annotate("{:d}|{:.2f}".format(labels[j], scores[j]),
                                pos,
                                fontsize=6,
                                color="b",
                                xytext=(pos[0] + 10, pos[1] - 10))

                    print("BBox params: {0}, score: {1}".format(boxes[j],
                                                                scores[j]))
                if image_save:
                    if image_save_suffix is not None:
                        save_name = "image_{0}_".format(i) + \
                                    image_save_suffix +  ".pdf"
                    else:
                        save_name = "image_{0}.pdf".format(i)

                    plt.savefig(image_save_dir + save_name,
                                dpi='figure', format='pdf')
                plt.show()

    results = {}
    if calc_f1:
        f1_list = eval_utils.calc_f1(tp_list, fp_list, fn_list)
        plot_utils.plot_f1(f1_list, confidence_values,
                           image_save=image_save,
                           image_save_dir=image_save_dir)
        results["F1"] = f1_list

    if calc_froc:
        froc_tpr, froc_fppi = eval_utils.calc_froc(tp_list, fp_list, fn_list)
        plot_utils.plot_frocs(froc_tpr, froc_fppi,
                              image_save=image_save,
                              image_save_dir=image_save_dir,
                              left_range=1e-2)
        results["FROC"] = {"TPR": froc_tpr, "FPPI": froc_fppi}

    if calc_ap:
        # ap = eval_utils.calc_ap(box_label_list, box_score_list)
        ap, precision_steps = eval_utils.calc_ap_MDT(box_label_list,
                                                     box_score_list,
                                                     num_gt)
        #print("Num_gt: {0}".format(num_gt))

        plot_utils.plot_precion_recall_curve(precision_steps,
                                             ap_value=ap,
                                             image_save=image_save,
                                             image_save_dir=image_save_dir,
                                             )
        print("AP: {0}".format(ap))

        prec, rec = eval_utils.calc_pr_values(box_label_list,
                                              box_score_list,
                                              num_gt)
        plot_utils.plot_precion_recall_curve(prec, rec,
                                             image_save=True,
                                             image_save_dir=image_save_dir,
                                             save_suffix="native")
        results["AP"] = {"AP": ap, "prec": prec, "rec": rec}

    if eval_class:
        # fpr, tpr, auroc = eval_utils.classification(image_class_list)
        # plot_utils.plot_roc(fpr, tpr, legend=[str(auroc)])
        cm, occ_classes = eval_utils.conf_matrix(image_class_list)
        plot_utils.plot_confusion_matrix(cm, classes=occ_classes,
                                         image_save=image_save,
                                         image_save_dir=image_save_dir)

    if image_save:
        # picle values for later (different) plotting
        with open(image_save_dir + "results", "wb") as result_file:
            pickle.dump(results, result_file)

if __name__ == '__main__':
    # paths to csv files containing labels (and other information)

    csv_calc_train = '/home/temp/moriz/data/' \
                     'calc_case_description_train_set.csv'
    csv_calc_test = '/home/temp/moriz/data/calc_case_description_test_set.csv'

    csv_mass_train = '/home/temp/moriz/data/' \
                     'mass_case_description_train_set.csv'
    csv_mass_test = '/home/temp/moriz/data/mass_case_description_test_set.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'

    # -----------------------------------------------------------------------
    # LOAD CHECKPOINT DIR CONTAINING MODEL FOR EVALIDATION
    # -----------------------------------------------------------------------

    # 2600x1300, bs=1, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-17_16-03-15")

    # 2600x1300, bs=1, lr=1e-5, unsegmented
    checkpoint_dir = get_paths("19-04-17_16-05-50")

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
    #checkpoint_dir = get_paths("19-04-18_20-29-55")

    # RN18, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-18_20-31-43")

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

    #-----------------------------------------------------------------------

    # RN18, IN, 4400x2200, bs=1, lr=1e-5, segmented
    #checkpoint_dir = get_paths("19-04-19_17-42-06")

    # RN18, IN, 1800x900, bs=1, lr=1e-5, segmented
    #checkpoint_dir = get_paths("19-04-19_18-17-01")

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-19_16-12-30")

    #-----------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-20_20-57-44")

    # RN50, IN, 1300x650, bs=1, lr=1e-6, unsegmented
    #checkpoint_dir = get_paths("19-04-20_20-59-20")

    #-----------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=2, lr=1e-4, unsegmented
    #checkpoint_dir = get_paths("19-04-20_21-11-42")

    # RN50, IN, 1800x900, bs=2, lr=1e-5, unsegmented
    #checkpoint_dir = get_paths("19-04-20_19-28-56")

    # RN50, IN, 1800x900, bs=2, lr=1e-6, unsegmented
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

    # -------------------------------------------------------------------------
    # RN50, IN, 2600x1300, bs=1, lr=1e-5, patho
    #checkpoint_dir = get_paths("19-04-21_21-00-57")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, birads
    #checkpoint_dir = get_paths("19-04-21_21-01-42")

    # -----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, seg
    #checkpoint_dir = get_paths("19-04-21_20-54-42")

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, seg
    #checkpoint_dir = get_paths("19-04-21_20-59-01")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, seg
    #checkpoint_dir = get_paths("19-04-21_21-06-40")

    # RN50, IN, 1300x650, bs=1, lr=1e-5, seg
    #checkpoint_dir = get_paths("19-04-21_21-07-38")

    #----------------------------------------------------------------------

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

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 1080Ti
    #checkpoint_dir = get_paths("19-04-23_21-57-17")

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 1080Ti, seg.
    #checkpoint_dir = get_paths("19-04-23_22-00-54")

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 980Ti
    #checkpoint_dir = get_paths("19-04-24_18-48-20")

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, seg.
    #checkpoint_dir = get_paths("19-04-24_18-49-34")

    #------------------------------------------------------------------------

    # INbreast RN50, 2600x1300 IN, bs=1, lr=1e-4,
    #checkpoint_dir = get_paths("19-04-24_18-55-11") + "run_04"

    # INbreast RN50, 2600x1300 IN, bs=1, lr=1e-5,
    #checkpoint_dir = get_paths("19-04-23_10-35-57") + "run_04"

    # INbreast RN50, 2600x1300 IN, bs=1, lr=1e-6,
    #checkpoint_dir = get_paths("19-04-24_18-54-06") + "run_04"

    # INbreast RN50, 2600x1300, bs=1, lr=1e-5, seg.
    #checkpoint_dir = get_paths("19-04-23_10-37-23") + "run_04"

    #------------------------------------------------------------------------

    # RN18, IN, 1800x900, bs=1, lr=1e-5, multistage test
    #checkpoint_dir = get_paths("19-04-26_20-59-29")

    # RN18, IN, 1800x900, bs=1, lr=1e-6, multistage test
    #checkpoint_dir = get_paths("19-04-26_21-13-41")

    #-----------------------------------------------------------------------

    # RN18, IN, 1800x900, bs=1, lr=1e-6, birads, softmax test
    #checkpoint_dir = get_paths("19-04-27_22-49-30")

    # RN50, IN, flex 980Ti, bs=1, lr=1e-5, seg test
    #checkpoint_dir = get_paths("19-04-28_13-52-53")

    # RN50, IN, flex 980Ti, bs=1, lr=1e-5, seg test
    # checkpoint_dir = get_paths("19-04-28_13-52-53")

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, better aug.
    #checkpoint_dir = get_paths("19-04-28_21-24-09")

    # RN50, IN, 2600x1300, one side flex, lr=1e-5, new seg. test
    #checkpoint_dir = get_paths("19-04-30_20-28-21")

    #-------------------------------------------------------------------------

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
    #checkpoint_dir = get_paths("19-04-30_19-38-27")

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

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, fix, better aug.
    #checkpoint_dir = get_paths("19-05-03_14-44-34")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, fix, seg., better aug.
    #checkpoint_dir = get_paths("19-05-03_14-45-20")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, fix, better aug.
    # checkpoint_dir = get_paths("19-05-03_15-16-15")

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex, better aug.
    # checkpoint_dir = get_paths("19-05-03_15-11-55")

    # ----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, patho.
    #checkpoint_dir = get_paths("19-05-03_23-55-21")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, birads
    #checkpoint_dir = get_paths("19-05-03_23-58-08")

    #-----------------------------------------------------------------------

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, flex, (baseline)
    #checkpoint_dir = get_paths("19-05-05_15-25-37")

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, flex, CL
    #checkpoint_dir = get_paths("19-05-01_11-25-14")

    # RN18, IN, 2600x1300, bs=1, lr=1e-6, flex, CL
    #checkpoint_dir = get_paths("19-05-01_11-25-55")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex, CL
    #checkpoint_dir = get_paths("19-05-01_11-22-10")

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, flex, CL
   # checkpoint_dir = get_paths("19-05-01_11-23-58")

    #------------------------------------------------------------------------

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, (baseline)
    #checkpoint_dir = get_paths("19-05-05_16-55-51")

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, CL
    #checkpoint_dir = get_paths("19-05-05_16-43-00")

    # RN18, IN, 2600x1300, bs=1, lr=1e-6, CL
    #checkpoint_dir = get_paths("19-05-05_16-47-51")

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, CL
    #checkpoint_dir = get_paths("19-05-05_16-51-09")

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, CL
    #checkpoint_dir = get_paths("19-05-05_16-52-08")


    #model_path = checkpoint_dir + "/checkpoint_best.pth"
    model_path = checkpoint_dir + "/checkpoint_epoch_25.pth"

    # flags and options
    settings = {"segment": False,
                "img_shape": [2600, 1300],
                #"shape_limit": [2600, 1300],
                #"shape_limit": "1080Ti",
                "norm": True,
                "type": "mass",
                # "crop_number": 9,
                "offset": 0,
                "num_elements": 20,
                "detection_only": True,
                "label_type": None,
                "resnet": "RN50",
                "set": "test",
                "NMS_thr": 0.2,
                }

    #kfold on INbreast
    # paths = inbreast_utils.get_paths(inbreast_dir,
    #                                  xls_file=xls_file,
    #                                  type="mass")
    #
    #
    # _, test_splits = utils.kfold_patientwise(paths,
    #                                          dataset_type="INbreast",
    #                                          num_splits=5,
    #                                          shuffle=True,
    #                                          random_state=42)
    #
    #
    # dataset = LazyINbreastDataset(inbreast_dir,
    #                               inbreast_utils.load_sample,
    #                               path_list = test_splits[4],
    #                               xls_file=xls_file,
    #                               **settings)

    # INbreast (for DDSM)
    dataset = LazyINbreastDataset(inbreast_dir,
                                  inbreast_utils.load_sample,
                                  xls_file=xls_file,
                                  **settings)

    # DDSM
    # dataset = LazyDDSMDataset(ddsm_dir,
    #                           ddsm_utils.load_sample,
    #                           #path_list=test_paths,
    #                           csv_file=csv_mass_test,
    #                           #csv_file=csv_calc_test,
    #                           **settings,
    #                           )


    main(dataset,
         model_path,
         plot_image=True,
         image_save=False,
         image_save_dir= "/home/temp/moriz/validation/final/test_results/",
         #image_save_subdir = "IoU_" + str(settings["NMS_thr"]),
         #image_save_subdir = "INbreast_IoU_" + str(settings["NMS_thr"]),
         calc_ap=False,
         calc_froc=False,
         calc_f1=False,
         eval_class=False,
         **settings)



