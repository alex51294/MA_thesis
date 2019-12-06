import torch
from tqdm import tqdm
import numpy as np
import os
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from detection.datasets import LazyDDSMDataset, LazyINbreastDataset
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors

from scipy.stats import norm

from scripts.paths import get_paths

def main(dataset,
         checkpoint_dir,
         start_epoch,
         end_epoch,
         step_size,
         results_save_dir = "/home/temp/moriz/validation/pickled_results/",
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

    # create dict were all results are saved
    total_results_dict = {}

    # get/ set crop_size
    if "crop_size" in settings and settings["crop_size"] is not None:
        crop_size = settings["crop_size"]
    else:
        crop_size = [600, 600]

    # determine used set
    if "set" in settings and settings["set"] is not None:
        set = settings["set"]
    else:
        raise KeyError("Missing set description!")

    if "resnet" not in settings.keys():
        resnet = "RN50"
    else:
        resnet = settings["resnet"]

    # set WBC factors
    crop_center = np.asarray(crop_size) / 2.0
    norm_pdf_var =  np.int32(min(crop_size[0], crop_size[1]) / 2. - 50)

    # create directory and file name
    cp_dir_date = checkpoint_dir.split("/")[-3]
    if "fold" not in settings:
        results_save_dir = results_save_dir + \
                           str(cp_dir_date) + "/" + \
                           set + \
                           "/image_level_" + \
                           str(start_epoch) + "_" + \
                           str(end_epoch) + "_" + \
                           str(step_size)
    else:
        results_save_dir = results_save_dir + \
                           str(cp_dir_date) + "/" + \
                           set + \
                           "/image_level_" + \
                           str(start_epoch) + "_" + \
                           str(end_epoch) + "_" + \
                           str(step_size) + \
                           "/fold_" + str(settings["fold"])


    # create folder (if necessary)
    if not os.path.isdir(results_save_dir):
        os.makedirs(results_save_dir)

    # gather all important settings in one dict and save them (pickle them)
    settings_dict = {"level": "image",
                     "checkpoint_dir": checkpoint_dir,
                     "start_epoch": start_epoch,
                     "end_epoch": end_epoch,
                     "step_size": step_size}
    settings_dict = {**settings_dict, **settings}

    with open(results_save_dir + "/settings", "wb") as settings_file:
        pickle.dump(settings_dict, settings_file)

    # iterate over the saved epochs and treat each epoch as separate model
    for epoch in tqdm(range(start_epoch, end_epoch + step_size, step_size)):
        checkpoint_path = checkpoint_dir + "/checkpoint_epoch_" + \
                          str(epoch) + ".pth"

        # load model
        checkpoint = torch.load(checkpoint_path)
        model = RetinaNet(**checkpoint['init_kwargs'], resnet=resnet).eval()
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.to(device)

        # dict for results from one model
        model_results_dict = {}

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                torch.cuda.empty_cache()

                # get image data
                test_data = dataset[i]

                # generate bboxes
                gt_bbox = utils.bounding_box(test_data["seg"])
                gt_bbox = torch.Tensor(gt_bbox).to(device)
                gt_label = test_data["label"]

                # generate crops
                crop_list, corner_list, heatmap =\
                    utils.create_crops(test_data,
                                       crop_size=crop_size,
                                       heatmap=True)

                # define list for predicted bboxes in crops
                image_bboxes = []
                image_scores = []
                image_labels = []
                crop_center_factor = []
                heatmap_factor = []

                # iterate over crops
                for j in range(0, len(crop_list)):
                    #CROP LEVEL
                    torch.cuda.empty_cache()
                    test_image = torch.Tensor(crop_list[j]['data']).to(device)

                    # predict anchors and labels for the crops using
                    # the loaded model
                    anchor_preds, cls_preds = model(test_image.unsqueeze(0))

                    # convert the predicted anchors to bboxes
                    anchors = Anchors()
                    boxes, labels, score = anchors.generateBoxesFromAnchors(
                        anchor_preds[0],
                        cls_preds[0],
                        (test_image.shape[2], test_image.shape[1]),
                        cls_tresh=0.05)

                    if boxes is None:
                        continue

                    # determine the center of each box and its distance to the
                    # crop center and calculate the resulting down-weighting
                    # factor based on it
                    box_centers = np.asarray(boxes[:, 0:2].to("cpu"))
                    dist = np.linalg.norm(crop_center - box_centers, ord=2,
                                          axis=1)
                    ccf = norm.pdf(dist, loc=0, scale=norm_pdf_var) * np.sqrt(
                        2 * np.pi) * norm_pdf_var

                    # the detected bboxes are relative to the crop; correct
                    # them with regard to the crop position in the image
                    for k in range(len(boxes)):
                        center_corrected = boxes[k][0:2] + \
                                           torch.Tensor(corner_list[j]).to(device)
                        image_bboxes.append(torch.cat((center_corrected,
                                                       boxes[k][2:])))
                        image_scores.append(score[k])
                        crop_center_factor.append(ccf[k])
                        image_labels.append(labels[k])

                # IMAGE LEVEL
                # determine heatmap factor based on the center posistion of
                # the bbox (required vor WBC only)
                for c in range(len(image_bboxes)):
                    pos_x = np.int32(image_bboxes[c][0].to("cpu"))
                    pos_x = np.minimum(np.maximum(pos_x, 0),
                                       test_data["data"].shape[2] - 1)

                    pos_y = np.int32(image_bboxes[c][1].to("cpu"))
                    pos_y = np.minimum(np.maximum(pos_y, 0),
                                       test_data["data"].shape[1] - 1)

                    heatmap_factor.append(heatmap[pos_y, pos_x])

                model_results_dict["image_%d" %i] = \
                    {"gt_list": gt_bbox,
                     "gt_label": gt_label,
                     "box_list": image_bboxes,
                     "score_list": image_scores,
                     "labels_list": image_labels,
                     "merging_utils": {"ccf": crop_center_factor,
                                       "hf": heatmap_factor}}


        # DATASET LEVEL
        total_results_dict[str(epoch)] = model_results_dict

    # MODELS LEVEL
    with open(results_save_dir + "/results", "wb") as result_file:
        torch.save(total_results_dict, result_file)


if __name__ == '__main__':
    # paths to csv files containing labels (and other information)
    csv_mass_train = \
        '/home/temp/moriz/data/mass_case_description_train_set.csv'
    csv_calc_train = \
        '/home/temp/moriz/data/calc_case_description_train_set.csv'
    csv_mass_all = '/home/temp/moriz/data/all_mass_cases.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'

    # path to image save directory
    image_save_dir = "/home/temp/moriz/validation/"

    #-----------------------------------------------------------------------
    # LOAD CHECKPOINT DIR CONTAINING EPOCHS FOR VALIDATION
    #-----------------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-21_20-44-17")

    # DDSM, RN34, IN, bs=1, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-24_12-29-08")

    #------------------------------------------------------------------------

    # DDSM, RN50, IN, bs=1, lr=1e-5, 600x600
    #checkpoint_dir = get_paths("19-04-19_12-59-19")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 900x900, broken
    #checkpoint_dir = get_paths("19-04-19_13-01-54")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 1200x1200
    #checkpoint_dir = get_paths("19-04-19_13-02-56")

    # DDSM, RN50, IN, bs=2, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-21_20-42-38")

    # DDSM, RN50, IN, bs=4, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-21_20-41-55")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 900x900, better aug.
    #checkpoint_dir = get_paths("19-04-21_21-16-26")

    # DDSM, RN50, IN, bs=1, lr=1e-6, 900x900
    #checkpoint_dir = get_paths("19-04-22_21-52-34")

    # DDSM, RN50, IN, bs=1, lr=1e-6, 900x900, better aug.
    #checkpoint_dir = get_paths("19-04-22_21-53-52")

    # DDSM, RN50, IN, bs=1, lr=1e-4, 900x900
    #checkpoint_dir = get_paths("19-04-22_21-55-18")

    # DDSM, RN50, IN, bs=1, lr=1e-4, 900x900, better aug.
    #checkpoint_dir = get_paths("19-04-22_21-54-47")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-24_12-28-33")

    #------------------------------------------------------------------------

    # DDSM, RN18, IN, bs=1, lr=1e-5, 900x900, new lazy mode
    # checkpoint_dir = get_paths("19-04-26_18-31-41")

    # DDSM, RN34, IN, bs=1, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-26-57")

    # DDSM, RN50, IN, bs=1, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-27_16-27-35")

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

    # ----------------------------------------------------------------------

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

    #--------------------------------------------------------------------

    # DDSM, RN18, IN, bs=2, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-36-27")

    # DDSM, RN18, IN, bs=4, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_19-30-10")

    # DDSM, RN18, IN, bs=1, lr=1e-5, 600x600, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-30-29")

    # DDSM, RN18, IN, bs=1, lr=1e-5, 1200x1200, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_18-34-13")

    # DDSM, RN18, BN, bs=4, lr=1e-5, 900x900, new lazy mode
    #checkpoint_dir = get_paths("19-04-26_19-02-14")

    # DDSM, RN18, BN, bs=8, lr=1e-5, 900x900, new lazy mode, broken
    #checkpoint_dir = get_paths("19-04-26_18-56-05")

    # ------------------------------------------------------------------

    # INbreast, CV, RN50, IN, bs=1, lr=1e-5, 600x600
    #checkpoint_dir = get_paths("19-04-23_10-39-22")

    # INbreast, CV, RN50, IN, bs=1, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-23_10-38-45")

    # INbreast, CV, RN50, IN, bs=1, lr=1e-5, 1200x1200
    #checkpoint_dir = get_paths("19-04-23_10-39-43")

    # INbreast, CV, RN50, IN, bs=1, lr=1e-4, 900x900
    #checkpoint_dir = get_paths("19-04-23_11-24-40")

    # INbreast, CV, RN50, IN, bs=1, lr=1e-6, 900x900
    #checkpoint_dir = get_paths("19-04-23_11-25-48")

    # INbreast, CV, RN18, IN, bs=1, lr=1e-5, 600x600
    #checkpoint_dir = get_paths("19-05-04_18-14-50")

    # INbreast, CV, RN18, IN, bs=1, lr=1e-5, 900x900
    #checkpoint_dir = get_paths("19-04-29_09-58-41")

    # INbreast, CV, RN18, IN, bs=1, lr=1e-5, 1200x1200
    #checkpoint_dir = get_paths("19-04-29_09-57-24")

    #------------------------------------------------------------------------

    # IN_RN18_1200x1200_bs_4_lr_1e-5
    checkpoint_dir = get_paths("19-04-29_22-34-07")

    # IN_RN18_1200x1200_bs_8_lr_1e-6
    #checkpoint_dir = get_paths("19-04-29_22-31-29")

    # IN_RN18_900x900_bs_8_lr_1e-5
    #checkpoint_dir = get_paths("19-05-02_19-11-02")

    # BN_RN18_900x900_bs_8_lr_1e-5
    #checkpoint_dir = get_paths("19-05-02_19-27-14")

    #------------------------------------------------------------------

    # IN_RN18_900x900_bs_1_lr_1e-4
    # checkpoint_dir = get_paths("19-05-02_12-13-43")

    # IN_RN18_900x900_bs_1_lr_1e-6
    #checkpoint_dir = get_paths("19-05-03_23-52-19")

    # IN_RN34_900x900_bs_1_lr_1e-4
    # checkpoint_dir = get_paths("19-05-02_12-09-44")

    # IN_RN34_900x900_bs_1_lr_1e-6
    # checkpoint_dir = get_paths("19-05-02_12-11-00")

    # IN_RN50_900x900_bs_1_lr_1e-4
    #checkpoint_dir = get_paths("19-05-03_00-14-58")

    # IN_RN50_900x900_bs_1_lr_1e-6
    #checkpoint_dir = get_paths("19-05-03_00-17-30")

    #---------------------------------------------------------------------
    # IN_RN18_600x600_bs_1_lr_1e-5, calc
    #checkpoint_dir = get_paths("19-05-03_00-12-59")

    # IN_RN18_1200x1200_bs_1_lr_1e-5, calc
    #checkpoint_dir = get_paths("19-05-03_00-10-53")

    # IN_RN18_1200x1200_bs_4_lr_1e-5, calc
    #checkpoint_dir = get_paths("19-05-07_15-04-32")

    # IN_RN18_1200x1200_bs_4_lr_1e-5, calc (repeated)
    #checkpoint_dir = get_paths("19-05-07_15-07-25")

    #------------------------------------------------------------------------
    #kfold
    #lr=1e-4, bs=2, [900, 900],
    # checkpoint_dir = "/home/temp/moriz/checkpoints/retinanet/" \
    #                  "inbreast/crop_based/" \
    #                  "inbreast_kfold_mass/19-04-07_14-48-22/checkpoints/"

    # lr=1e-4, bs=4, [600, 600],
    # checkpoint_dir = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "crop_based/" \
    #                  "inbreast_kfold_mass/19-04-07_14-31-24/checkpoints/"


    # flags and options for validation; must not but should coincidence with
    # the settings of the training (see config file or above annotations)

    settings = {"segment": False,
                "crop_size": [1200, 1200],
                "img_shape": None,
                "random_seed": 42,
                "set": "val",
                "norm": True,
                "type": "calc",
                "detection_only": True,
                "num_elements": None,
                #"label_type": "pathology",
                "resnet": "RN18",
                }

    # INbreast
    # _, _, val_paths = inbreast_utils.load_single_set(inbreast_dir,
    #                                                  xls_file=xls_file,
    #                                                  train_size=0.7,
    #                                                  val_size=0.15,
    #                                                  type=settings["type"],
    #                                                  random_state=settings["random_seed"])
    #
    # # load INbreast data
    # dataset = LazyINbreastDataset(inbreast_dir,
    #                               inbreast_utils.load_sample,
    #                               path_list = val_paths,
    #                               xls_file=xls_file,
    #                               **settings)

    # DDSM
    _, val_paths, _ = ddsm_utils.load_single_set(ddsm_dir,
                                                 #csv_file= csv_mass_train,
                                                 csv_file= csv_calc_train,
                                                 train_size=0.9,
                                                 val_size=None,
                                                 random_state=settings["random_seed"])

    dataset = LazyDDSMDataset(ddsm_dir,
                              ddsm_utils.load_sample,
                              path_list= val_paths,
                              #csv_file=csv_mass_train,
                              csv_file=csv_calc_train,
                              **settings)

    main(dataset,
         checkpoint_dir = checkpoint_dir,
         start_epoch=4,
         end_epoch=24,
         step_size=4,
         results_save_dir="/home/temp/moriz/validation/results/",
         **settings)

    #kfold
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
    #          results_save_dir="/home/temp/moriz/validation/final/results/",
    #          **settings)
