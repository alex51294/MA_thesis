import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

from tqdm import tqdm
import numpy as np
import os
import pickle

from detection.datasets import LazyINbreastDataset, LazyDDSMDataset
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors

import scripts.eval_utils as eval_utils
import scripts.plot_utils as plot_utils
from scripts.paths import get_paths

from scipy.stats import norm

# TODO:find out WHY IN THE HELL THIS IS REQUIRED
#matplotlib.use('module://backend_interagg')

def main(dataset,
         model_path,
         plot_image=False,
         plot_crops=False,
         image_save=False,
         image_save_dir=None,
         image_save_subdir=None,
         image_save_suffix=None,
         calc_ap=False,
         calc_froc=False,
         calc_f1=False,
         thr=0.5,
         **settings):

    # device
    device = 'cuda'

    # get/ set crop_size
    if "crop_size" in settings and settings["crop_size"] is not None:
        crop_size = settings["crop_size"]
    else:
        crop_size = [600, 600]

    # get/ set crop_size
    if "merging_method" in settings and settings["merging_method"] is not None:
        merging_method = settings["merging_method"]
    else:
        merging_method = "NMS"

    if "resnet" not in settings.keys():
        resnet = "RN50"
    else:
        resnet = settings["resnet"]

    # load model
    checkpoint = torch.load(model_path)
    model = RetinaNet(**checkpoint['init_kwargs'], resnet=resnet)
    model.eval()
    model.load_state_dict(checkpoint['state_dict']["model"])
    #model.load_state_dict(checkpoint['state_dict'])
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
            image_save_dir += image_save_subdir + "/"

        if not os.path.isdir(image_save_dir):
            os.makedirs(image_save_dir)

    tp_list = []
    fp_list = []
    fn_list = []
    iou_list = []

    confidence_values = np.arange(0.05, 1, 0.05)

    box_label_list = []
    box_score_list = []

    if plot_image and plot_crops:
        plot_crops = False

    # set WBC factors
    crop_center = np.asarray(crop_size) / 2.0
    norm_pdf_var = np.int32(min(crop_size[0], crop_size[1]) / 2. - 50)

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            torch.cuda.empty_cache()

            # get image data
            test_data = dataset[i]

            # load GT
            gt_bbox = utils.bounding_box(test_data["seg"])
            gt_bbox = torch.Tensor(gt_bbox).to(device)
            gt_label = test_data["label"]

            # generate crops
            crop_list, corner_list, heatmap = \
                utils.create_crops(test_data,
                                   crop_size=crop_size,
                                   overlap=0.5,
                                   heatmap=True)

            # define list for predicted bboxes in crops
            crop_bbox = []
            score_bbox = []
            class_bbox = []
            crop_center_factor = []
            heatmap_factor = []

            # iterate over crops
            for j in tqdm(range(0, len(crop_list))):
                #CROP LEVEL
                torch.cuda.empty_cache()
                test_image = torch.Tensor(crop_list[j]['data']).to(device)
                # test_image = crop_list[j]['data']
                # test_image = (test_image - np.mean(test_image)) / (np.std(test_image) + 1e-8)
                # test_image = torch.Tensor(test_image).to(device)

                crop_gt_bbox = utils.bounding_box(crop_list[j]['seg'])

                # predict anchors and labels for the crops using the loaded model
                anchor_preds, cls_preds = model(test_image.unsqueeze(0))

                # convert the predicted anchors to bboxes
                anchors = Anchors()
                boxes, labels, score = anchors.generateBoxesFromAnchors(
                    anchor_preds[0],
                    cls_preds[0],
                    (test_image.shape[2], test_image.shape[1]),
                    cls_tresh=0.05)

                # plot the results on the single crops (if desired)
                if plot_crops:
                    # plot the single crops
                    plt.figure(1, figsize=(15, 10))
                    fig, ax = plt.subplots(1)

                    ax.imshow(test_image[0, :, :], cmap='Greys_r')
                    # show bboxes as saved in data (in red with center)
                    for l in range(len(crop_gt_bbox)):
                        pos = tuple(crop_gt_bbox[l][0:2])
                        plt.plot(pos[0], pos[1], 'r.')
                        width = crop_gt_bbox[l][2]
                        height = crop_gt_bbox[l][3]
                        pos = (pos[0] - torch.floor(width / 2),
                               pos[1] - torch.floor(height / 2))

                        # Create a Rectangle patch
                        rect = patches.Rectangle(pos, width, height,
                                                 linewidth=1,
                                                 edgecolor='r',
                                                 facecolor='none')
                        ax.add_patch(rect)

                    #show the predicted bboxes in the single crops (in blue)
                    # keep = score > 0.5
                    # boxes = boxes[keep]
                    # score = score[keep]
                    for l in range(len(boxes)):
                        width = boxes[l][2]
                        height = boxes[l][3]
                        pos = (boxes[l][0] - torch.floor(width/2),
                               boxes[l][1] - torch.floor(height/2))

                        # Create a Rectangle patch
                        rect = patches.Rectangle(pos, width, height, linewidth=1,
                                                 edgecolor='b', facecolor='none')
                        ax.add_patch(rect)
                        ax.annotate("{:.2f}".format(score[l]), pos,
                                                    fontsize=6,
                                                    color="b",
                                                    xytext=(pos[0]+10, pos[1]-10))
                    plt.show()

                # TODO: discuss whether appropriate
                if boxes is None:
                    continue

                # determine the center of each box and its distance to the
                # crop center and calculate the resulting down-weighting factor
                # based on it
                box_centers = np.asarray(boxes[:, 0:2].to("cpu"))
                dist = np.linalg.norm(crop_center-box_centers, ord=2, axis=1)
                ccf = norm.pdf(dist, loc=0, scale=norm_pdf_var) * \
                      np.sqrt(2*np.pi) * norm_pdf_var

                # correct the predicted bboxes and save all values into
                # according lists
                for k in range(len(boxes)):
                    center_corrected = boxes[k][0:2] + \
                                       torch.Tensor(corner_list[j]).to(device)
                    crop_bbox.append(torch.cat((center_corrected,
                                               boxes[k][2:])))
                    score_bbox.append(score[k])
                    class_bbox.append(labels[k])
                    crop_center_factor.append(ccf[k])

            # IMAGE LEVEL
            # determine heatmap factor based on the center posistion of
            # the bbox
            for c in range(len(crop_bbox)):
                pos_x = np.int32(crop_bbox[c][0].to("cpu"))
                pos_x = np.minimum(np.maximum(pos_x, 0), test_data["data"].shape[2]-1)

                pos_y = np.int32(crop_bbox[c][1].to("cpu"))
                pos_y = np.minimum(np.maximum(pos_y, 0), test_data["data"].shape[1]-1)

                heatmap_factor.append(heatmap[pos_y, pos_x])


            # merge crop-level predictions
            if len(crop_bbox) > 0:
                # merge overlapping bounding boxes using NMS
                if merging_method == "NMS":
                    image_bbox, score_bbox, class_bbox = \
                        eval_utils.nms(crop_bbox, score_bbox, class_bbox, thr=thr)

                # merge overlapping bounding boxes using WBC
                elif merging_method == "WBC":

                    #image_bbox, score_bbox = eval_utils.wbc(crop_bbox, score_bbox, 0.2)

                    #merge overlapping bounding boxes using my merging
                    image_bbox, score_bbox = eval_utils.my_merging(crop_bbox,
                                                                   score_bbox,
                                                                   crop_center_factor,
                                                                   heatmap_factor,
                                                                   thr=thr)
                    # image_bbox, score_bbox, _ = \
                    #     eval_utils.rm_overlapping_boxes(image_bbox,
                    #                                     score_bbox,
                    #                                     order="xywh")

                elif merging_method == "Jung":
                    image_bbox, score_bbox = eval_utils.merge_jung(crop_bbox,
                                                                   score_bbox,
                                                                   merge_thr=0.2)


                # calculate the required rates for the FROC metric
                tp_crop, fp_crop, fn_crop = \
                    eval_utils.calc_tp_fn_fp(gt_bbox,
                                             image_bbox,
                                             score_bbox,
                                             confidence_values=confidence_values)
                tp_list.append(tp_crop)
                fp_list.append(fp_crop)
                fn_list.append(fn_crop)

                # determine the overlap of detected bbox with the ground truth
                box_label, box_score, _ = \
                    eval_utils.calc_detection_hits(gt_bbox,
                                                   image_bbox,
                                                   score_bbox,
                                                   score_thr=0.0)
                box_label_list.append(box_label)
                box_score_list.append(box_score)
            else:
                image_bbox = []
                tp_list.append([torch.tensor(0, device=device) for tp
                                in range(len(confidence_values))])
                fp_list.append([torch.tensor(0, device=device) for fp
                                in range(len(confidence_values))])
                fn_list.append([torch.tensor(1, device=device) for fn
                                in range(len(confidence_values))])

            # plot the results on the single images (if desired)
            if plot_image:
                # plot image
                c, h, w = test_data["data"].shape
                figsize = 0.25 * (w / 100), 0.25 * (h / 100)
                fig, ax = plt.subplots(1, figsize=figsize)

                ax.imshow(test_data["data"][0, :, :], cmap='Greys_r')
                gt_bbox = gt_bbox.to("cpu")

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
                    ax.annotate("{:d}".format(np.int32(gt_label)),
                                pos,
                                fontsize=6,
                                color="r",
                                xytext=(pos[0] + 10, pos[1] - 10))

                # show the predicted bboxes (in blue)
                print("Number of detected bboxes: {0}".format(len(image_bbox)))
                # keep = score_bbox > 0.3
                # image_bbox = image_bbox[keep]
                # score_bbox = score_bbox[keep]
                for j in range(len(image_bbox)):
                    width = image_bbox[j][2]
                    height = image_bbox[j][3]
                    pos = (image_bbox[j][0] - torch.floor(width / 2),
                           image_bbox[j][1] - torch.floor(height / 2))

                    # Create a Rectangle patch
                    rect = patches.Rectangle(pos, width, height, linewidth=1,
                                             edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    ax.annotate("{:d}: {:.2f}".format(class_bbox[j], score_bbox[j]),
                                pos,
                                fontsize=6,
                                color="b",
                                xytext=(pos[0]+10, pos[1]-10))

                    print("BBox params: {0}, score: {1}".format(image_bbox[j],
                                                                score_bbox[j]))
                if image_save:
                    if image_save_suffix is not None:
                        save_name = "image_{0}_".format(i) + \
                                    image_save_suffix +  ".pdf"
                    else:
                        save_name = "image_{0}.pdf".format(i)

                    plt.savefig(image_save_dir + save_name,
                                dpi='figure', format='pdf')
                plt.show()

    # DATASET LEVEL
    results = {}
    if calc_ap:
        #ap = eval_utils.calc_ap(box_label_list, box_score_list)
        ap, precision_steps = eval_utils.calc_ap_MDT(box_label_list,
                                                     box_score_list,
                                                     len(dataset))
        plot_utils.plot_precion_recall_curve(precision_steps,
                                             ap_value = ap,
                                             image_save=image_save,
                                             image_save_dir=image_save_dir,
                                             save_suffix=merging_method,
                                             )
        print("AP: {0}".format(ap))

        prec, rec = eval_utils.calc_pr_values(box_label_list,
                                              box_score_list,
                                              len(dataset))
        plot_utils.plot_precion_recall_curve(prec, rec)

        results["AP"] = {"AP":ap, "prec":prec, "rec":rec}


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

    if image_save:
        # picle values for later (different) plotting
        with open(image_save_dir + "results", "wb") as result_file:
            pickle.dump(results, result_file)

if __name__ == '__main__':
    # paths to csv files containing labels (and other information)
    csv_mass_train = \
        '/home/temp/moriz/data/mass_case_description_train_set.csv'
    csv_mass_test = \
        '/home/temp/moriz/data/mass_case_description_test_set.csv'
    csv_calc_test = \
        '/home/temp/moriz/data/calc_case_description_test_set.csv'

    csv_mass_all = '/home/temp/moriz/data/all_mass_cases.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'



    # -----------------------------------------------------------------------
    # LOAD CHECKPOINT DIR CONTAINING MODEL FOR EVALUATION
    # -----------------------------------------------------------------------

    # DDSM, IN, 600x600, n_train=all, n_val=all
    #checkpoint_dir = get_paths("19-04-19_12-59-19")

    # DDSM, IN, 900x900, n_train=all, n_val=all, broken
    #checkpoint_dir = get_paths("19-04-19_13-01-54")

    #DDSM, IN, 1200x1200, n_train = all, n_val = all
    #checkpoint_dir = get_paths("19-04-19_13-02-56")

    # DDSM, RN50, IN, bs=2, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-21_20-42-38")

    # DDSM, RN50, IN, bs=4, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-21_20-41-55")

    #--------------------------------------------------------------------------

    # DDSM, RN50, IN, bs=1, lr=1e-4, 900x900
    # checkpoint_dir = get_paths("19-04-22_21-55-18")

    # DDSM, RN50 IN, bs=1, lr=1e-5, 900x900,
    #checkpoint_dir = get_paths("19-04-24_12-28-33")

    # DDSM, RN50, IN, bs=1, lr=1e-6, 900x900
    #checkpoint_dir = get_paths("19-04-22_21-52-34")

    #---------------------------------------------------------------------

    # DDSM, RN50, IN, bs=1, lr=1e-4, 900x900, better aug.
    #checkpoint_dir = get_paths("19-04-22_21-54-47")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 900x900, better aug.
    # checkpoint_dir = get_paths("19-04-21_21-16-26")

    # DDSM, RN50, IN, bs=1, lr=1e-6, 900x900, better aug.
    #checkpoint_dir = get_paths("19-04-22_21-53-52")

    #---------------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-5, 900x900, cached
    #checkpoint_dir = get_paths("19-04-21_20-44-17")

    # DDSM, RN34, IN, bs=1, lr=1e-5, 900x900, cached
    #checkpoint_dir = get_paths("19-04-24_12-29-08")

    #----------------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-31-41")

    # DDSM, RN34, IN, bs=1, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-26-57")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-27-35")

    #----------------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-5, 600x600, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-30-29")

    # DDSM, RN18, IN, bs=1, lr=1e-5, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-34-13")

    #----------------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-6, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_17-01-16")

    # DDSM, RN18, IN, bs=2, lr=1e-5, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-58-38")

    # DDSM, RN18, IN, bs=2, lr=1e-6, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-59-57")

    # DDSM, RN18, IN, bs=4, lr=1e-5, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-56-50")

    # DDSM, RN18, IN, bs=4, lr=1e-6, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-55-51")

    # DDSM, RN18, IN, bs=8, lr=1e-5, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-50-45")

    # DDSM, RN18, IN, bs=8, lr=1e-6, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-52-23")

    # DDSM, RN18, BN, bs=8, lr=1e-5, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_23-11-34")

    # DDSM, RN18, BN, bs=8, lr=1e-6, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_23-15-01")

    #-----------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-5, 900x900, new lazy mode, better aug.
    #checkpoint_dir = get_paths("19-05-04_17-11-20")

    # DDSM, RN18, IN, bs=1, lr=1e-5, 1200x1200, new lazy mode, better aug.
    #checkpoint_dir = get_paths("19-04-27_23-00-22")

    # DDSM, RN18, IN, bs=1, lr=1e-6, 1200x1200, new lazy mode, better aug.
    #checkpoint_dir = get_paths("19-04-27_23-01-58")

    # DDSM, RN18, IN, bs=1, lr=1e-6, 1200x1200, new lazy mode, better aug.
    #checkpoint_dir = get_paths("19-05-04_18-10-38")

    # IN_RN18_1200x1200_bs_4_lr_1e-5_better_aug
    # checkpoint_dir = get_paths("19-04-29_22-27-26")

    # IN_RN18_1200x1200_bs_4_lr_1e-5_better_aug
    #checkpoint_dir = get_paths("19-05-04_17-13-12")

    #-------------------------------------------------------------------

    # DDSM, RN18, IN, bs=2, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-36-27")

    # DDSM, RN18, IN, bs=4, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_19-30-10")

    # DDSM, RN18, BN, bs=4, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_19-02-14")

    # DDSM, RN18, BN, bs=8, lr=1e-5, 900x900, new lazy mode, broken
    #checkpoint_dir = get_paths("19-04-26_18-56-05")

    #-----------------------------------------------------------------------
    # INbreast, CV, RN50, IN, bs=1, lr=1e-5, 600x600
    #checkpoint_dir = get_paths("19-04-23_10-39-22") + "run_04"

    # INbreast, CV, RN50, IN, bs=1, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-23_10-38-45") + "run_04"

    # INbreast, CV, RN50, IN, bs=1, lr=1e-5, 1200x1200
    #checkpoint_dir = get_paths("19-04-23_10-39-43") + "run_04"

    # INbreast, CV, RN50, IN, bs=1, lr=1e-4, 900x900
    #checkpoint_dir = get_paths("19-04-23_11-24-40") + "run_04"

    # INbreast, CV, RN50, IN, bs=1, lr=1e-6, 900x900
    #checkpoint_dir = get_paths("19-04-23_11-25-48") + "run_04"

    # INbreast, CV, RN18, IN, bs=1, lr=1e-5, 600x600
    #checkpoint_dir = get_paths("19-05-04_18-14-50") + "run_04"

    # INbreast, CV, RN18, IN, bs=1, lr=1e-4, 900x900
    #checkpoint_dir = get_paths("19-04-29_09-58-41") + "run_04"

    # INbreast, CV, RN18, IN, bs=1, lr=1e-5, 1200x1200
    #checkpoint_dir = get_paths("19-04-29_09-57-24") + "run_04"

    #-----------------------------------------------------------------------

    # IN_RN18_1200x1200_bs_4_lr_1e-5
    #checkpoint_dir = get_paths("19-04-29_22-34-07")

    # IN_RN18_1200x1200_bs_4_lr_1e-5_better_aug
    #checkpoint_dir = get_paths("19-04-29_22-27-26")

    # IN_RN18_1200x1200_bs_8_lr_1e-6
    #checkpoint_dir = get_paths("19-04-29_22-31-29")

    #-----------------------------------------------------------------------

    # checkpoint_dir = "/home/temp/moriz/checkpoints/retinanet/" \
    #                  "ddsm/ddsm_final/" \
    #                  "ddsm_cb_mass_IN_RN18_900x900_lr_1e-4/" \
    #                  "19-05-02_12-13-43/checkpoints/run_00"

    # checkpoint_dir = "/home/temp/moriz/checkpoints/retinanet/" \
    #                  "ddsm/ddsm_final/" \
    #                  "ddsm_cb_mass_IN_RN34_900x900_lr_1e-4/" \
    #                  "19-05-02_12-09-44/checkpoints/run_00"

    # checkpoint_dir = "/home/temp/moriz/checkpoints/retinanet/" \
    #                  "ddsm/ddsm_final/" \
    #                  "ddsm_cb_mass_IN_RN34_900x900_lr_1e-6/" \
    #                  "19-05-02_12-11-00/checkpoints/run_00"

    # IN_RN18_600x600_bs_1_lr_1e-5, calc
    # checkpoint_dir = get_paths("19-05-03_00-12-59")

    # IN_RN18_1200x1200_bs_1_lr_1e-5, calc
    # checkpoint_dir = get_paths("19-05-03_00-10-53")

    # IN_RN18_600x600_bs_1_lr_1e-5, calc
    # checkpoint_dir = get_paths("19-05-03_00-12-59")

    # IN_RN18_1200x1200_bs_4_lr_1e-5, calc
    checkpoint_dir = get_paths("19-05-07_15-04-32")

    # IN_RN18_1200x1200_bs_1_lr_1e-5, calc
    #checkpoint_dir = get_paths("19-05-03_00-10-53")

    # IN_RN18_900x900_bs_1_lr_1e-6
    #checkpoint_dir = get_paths("19-05-03_23-52-19")

    # IN_RN50_900x900_bs_1_lr_1e-4
    #checkpoint_dir = get_paths("19-05-03_00-14-58")

    # IN_RN50_900x900_bs_1_lr_1e-6
    #checkpoint_dir = get_paths("19-05-03_00-17-30")

    # IN_RN18_900x900_bs_8_lr_1e-5
    #checkpoint_dir = get_paths("19-05-02_19-11-02")

    # BN_RN18_900x900_bs_8_lr_1e-5
    #checkpoint_dir = get_paths("19-05-02_19-27-14")

    #model_path = checkpoint_dir + "/checkpoint_best.pth"
    model_path = checkpoint_dir + "/checkpoint_epoch_48.pth"

    # ------------------------------------------------------------------------

    # flags and options
    settings = {"segment": False,
                "crop_size": [1200, 1200],
                "img_shape": None,
                "random_seed": 42,
                "set": "val",
                "norm": True,
                "type": "calc",
                "detection_only": True,
                "offset": 0,
                "num_elements": None,
                # "label_type": "pathology",
                "resnet": "RN18",
                "merging_method": "NMS",
                "thr": 0.2,
                }

    #kfold
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



    dataset = LazyDDSMDataset(ddsm_dir,
                              ddsm_utils.load_sample,
                              #csv_file=csv_mass_test,
                              csv_file=csv_calc_test,
                              **settings
                              )

    # dataset = LazyINbreastDataset(inbreast_dir,
    #                               inbreast_utils.load_sample,
    #                               xls_file=xls_file,
    #                               **settings)

    main(dataset,
         model_path,
         plot_image=False,
         plot_crops=False,
         image_save=False,
         image_save_dir="/home/temp/moriz/validation/final/test_results/",
         # image_save_subdir= settings["merging_method"] + "_" +
         #                    str(settings["thr"]),
         #image_save_subdir= "INbreast",
         calc_ap=True,
         calc_froc=True,
         calc_f1=True,
         **settings)