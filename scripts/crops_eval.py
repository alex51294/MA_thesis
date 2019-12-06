import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import os
import pickle

from detection.datasets import LazyDDSMDataset, CacheINbreastDataset, \
    LazyINbreastDataset
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors


# # TODO:find out WHY IN THE HELL THIS IS REQUIRED
# if matplotlib.get_backend() == "agg":
#     matplotlib.use('module://backend_interagg')


def main(dataset,
         checkpoint_dir,
         start_epoch,
         end_epoch,
         step_size,
         results_save_dir = "/home/temp/moriz/validation/pickled_results/",
         plot=False,
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

    # set WBC factors
    crop_center = np.asarray(crop_size) / 2.0
    norm_pdf_var = np.int32(min(crop_size[0], crop_size[1]) / 2. - 50)

    # create directory and file name
    cp_dir_date = checkpoint_dir.split("/")[-3]
    results_save_dir = results_save_dir + str(cp_dir_date) + "/" + set + \
                       "/crop_level_" + str(start_epoch) + \
                       "_" + str(end_epoch) + "_" + str(step_size)

    # create folder (if necessary)
    if not os.path.isdir(results_save_dir):
        os.makedirs(results_save_dir)

    # gather all important settings in one dict and save them (pickle them)
    settings_dict = {"level": "crops",
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
        model = RetinaNet(**checkpoint['init_kwargs']).eval()
        model.load_state_dict(checkpoint['state_dict']["model"])
        #model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        # dict for results from one model
        model_results_dict = {}

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                torch.cuda.empty_cache()

                # get image data
                data = dataset[i]

                # iterate over the number of crops generated from image i
                for k in range(len(data)):
                    crop_data = data[k]
                    gt_bbox = utils.bounding_box(crop_data["seg"])

                    torch.cuda.empty_cache()
                    crop = torch.Tensor(crop_data['data']).to(device)
                    gt_bbox = torch.Tensor(gt_bbox).to(device)
                    gt_label = crop_data["label"]

                    # predict anchors and labels for the crops using the loaded model
                    anchor_preds, cls_preds = model(crop.unsqueeze(0))

                    # convert the predicted anchors to bboxes
                    anchors = Anchors()

                    boxes, labels, scores = \
                        anchors.generateBoxesFromAnchors(anchor_preds[0],
                                                         cls_preds[0],
                                                         (crop.shape[2],
                                                          crop.shape[1]),
                                                         cls_tresh=0.05)

                    # show the predicted bboxes (in blue)
                    # plot the image with the according bboxes
                    if plot:
                        gt_bbox = gt_bbox.to("cpu")

                        # plot image
                        c, h, w = crop_data["data"].shape
                        figsize = (w / 100), (h / 100)
                        fig, ax = plt.subplots(1, figsize=figsize)

                        ax.imshow(crop_data["data"][0, :, :], cmap='Greys_r')

                        # show bboxes as saved in data (in red with center)
                        for l in range(len(gt_bbox)):
                            pos = tuple(gt_bbox[l][0:2])
                            plt.plot(pos[0], pos[1], 'r.')
                            width = gt_bbox[l][2]
                            height = gt_bbox[l][3]
                            pos = (pos[0] - np.floor(width / 2),
                                   pos[1] - np.floor(height / 2))

                            # Create a Rectangle patch
                            rect = patches.Rectangle(pos, width, height,
                                                     linewidth=1,
                                                     edgecolor='r',
                                                     facecolor='none')
                            ax.add_patch(rect)

                        # keep = score_bbox > 0.5
                        # crop_boxes = crop_boxes[keep]
                        # score_bbox = score_bbox[keep]
                        for j in range(len(boxes)):
                            width = boxes[j][2]
                            height = boxes[j][3]
                            pos = (boxes[j][0] - torch.floor(width / 2),
                                   boxes[j][1] - torch.floor(height / 2))

                            # Create a Rectangle patch
                            rect = patches.Rectangle(pos, width, height, linewidth=1,
                                                     edgecolor='b', facecolor='none')
                            ax.add_patch(rect)
                            ax.annotate("{:.2f}".format(scores[j]), pos,
                                        fontsize=10,
                                        color="b",
                                        xytext=(pos[0]+10, pos[1]-10))
                        plt.show()

                    model_results_dict["image_{0}_crop_{1}".format(i, k)] = \
                        {"gt_list": gt_bbox,
                         "gt_label": gt_label,
                         "box_list": boxes,
                         "score_list": scores}

        # DATASET LEVEL
        total_results_dict[str(epoch)] = model_results_dict


    # MODELS LEVEL
    with open(results_save_dir + "/results", "wb") as result_file:
        torch.save(total_results_dict, result_file)


if __name__ == '__main__':
    # paths to csv files containing labels (and other information)
    csv_mass_train = \
        '/home/temp/moriz/data/mass_case_description_train_set.csv'

    csv_mass_all = '/home/temp/moriz/data/all_mass_cases.csv'

    # path to data directory
    ddsm_dir = '/home/temp/moriz/data/CBIS-DDSM/'

    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'


    # Adam training with bs=4, lr=1-e4 and reduced, e_max = 1500, new augmentation  (rot90)
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-25_14-36-40/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e5 and reduced, e_max = 1500, new augmentation  (rot90)
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-25_14-38-31/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e6 and reduced, e_max =1500, new augmentation  (rot90)
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-25_14-43-54/checkpoints/run_00"

    # Adam training with bs=8, lr=1-e6 and reduced, e_max =1500, new augmentation  (rot90)
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-26_16-26-36/checkpoints/run_00"

    # Adam training with bs=8, lr=1e-5, e_max = 500, new aug
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-29_12-46-44/checkpoints/run_00"

    # Adam training with bs=8, lr=1e-4, e_max = 1500, new aug
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-29_14-59-19/checkpoints/run_00"

    # Adam training with bs=8, lr=1-e6 e_max = 2500, new aug
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-31_09-41-11/checkpoints/run_00"

    # Adam training with bs=8, lr=1-e7 e_max = 2500, new aug
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-01-31_09-50-48/checkpoints/run_00"

    #Adam training with bs=4, lr=1-e4, e_max = 1000, new aug, s=42
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-02-02_20-11-53/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 1000, new aug, s=5
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-02-02_20-09-40/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 1500, new aug, s=42,
    # crop_size = [800, 800]
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "19-02-04_14-33-35/checkpoints/run_00"

    #------------------------------------------------------------------------

    # Adam training with bs=4, lr=1-e4, e_max = 750, new aug, s=42,
    # crop_size = [750, 750]
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #              "inbreast_lr_1e-4_b_4_e_750_cs_750_750/" \
    #              "19-02-15_19-34-23/checkpoints/run_00"

    # DDSM: Adam training with bs=8, lr=1-e6, e_max = 1500, new aug, s=42,
    # crop_size = [600, 600]
    # model_path = "/home/temp/moriz/checkpoints/retinanet/ddsm/" \
    #                  "ddsm_lr_1e-6_b_8_e_1500_n_76_s_42/" \
    #                  "19-02-14_21-07-48/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 1500, new aug, s=42,
    # crop_size = [1200, 700]
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_lr_1e-4_b_4_e_1500_cs_1200_700/" \
    #                  "19-02-17_20-12-18/checkpoints/run_00"

    # DDSM: Adam training with bs=8, lr=1-e4, e_max = 1500, new aug, s=42,
    # crop_size = [600, 600]
    # model_path = "/work/scratch/moriz/checkpoints/retinanet/ddsm/" \
    #              "ddsm_lr_1e-4_b_8_e_1500_n_76_cs_600_600/" \
    #              "19-02-17_19-04-07/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 1500, new aug, s=42,
    # crop_size = [900, 900], unsegmented
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_lr_1e-4_b_4_e_1500_cs_900_900_unsegmented/" \
    #                  "19-02-20_12-49-45/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 1500, new aug, s=42,
    # crop_size = [1300, 1300], unsegmented
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_lr_1e-4_b_2_e_1500_cs_1300_1300_unsegmented/" \
    #                  "19-02-20_18-22-54/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e5, e_max = 1500, new aug, s=42,
    # crop_size = [1300, 1300], unsegmented
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_lr_1e-5_b_2_e_1500_cs_1300_1300_unsegmented/" \
    #                  "19-02-20_20-48-28/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 500, extensive aug, s=42,
    # crop_size = [900, 900], unsegmented
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_lr_1e-4_b_4_e_500_cs_900_900_unsegmented_better_aug/" \
    #                  "19-03-04_19-25-48/checkpoints/run_00"


    # Adam training with bs=4, lr=1-e2, e_max = 500, extensive aug, s=42,
    # crop_size = [1300, 650] and whole samples, segmented
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_sample_&_crops_lr_1e-4_b_2_e_500_cs_1300_650_segmented/" \
    #                  "19-03-08_20-33-47/checkpoints/run_00"

    # Adam training with bs=4, lr=1-e4, e_max = 500, extensive aug, s=42,
    # crop_size = [1300, 650], unsegmented
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_lr_1e-4_b_4_e_500_cs_1300_650_unsegmented/" \
    #                  "19-03-09_16-36-29/checkpoints/run_00"


    #-----------------------------------------------------------------------

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_TEST_2_lr_1e-4_b_4_e_500_cs_900_900/" \
    #                  "19-03-22_10-28-18/checkpoints/run_00"

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
    #                  "inbreast_TEST_3_lr_1e-4_b_4_e_500_cs_900_900/" \
    #                  "19-03-22_13-40-03/checkpoints/run_00"

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-4_b_4_e_500_cs_600_600/" \
    #                  "19-03-22_21-07-54/checkpoints/run_00"

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-5_b_4_e_500_cs_600_600/" \
    #                  "19-03-24_01-05-12/checkpoints/run_00"

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-6_b_4_e_500_cs_600_600/" \
    #                  "19-03-23_20-12-41/checkpoints/run_00"

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_birads_lr_1e-4_b_4_e_500_cs_900_900/" \
    #                  "19-03-25_04-57-57/checkpoints/run_00"

    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/calc/" \
    #                  "inbreast_lr_1e-4_b_4_e_500_cs_600_600/" \
    #                  "19-03-27_18-02-31/checkpoints/run_00"

    # lr=1e-4, bs=1, [900, 900], IN
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/test/" \
    #                  "19-03-29_01-39-12/checkpoints/run_00"

    # lr=1e-4, bs=1, [900, 900], BN
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/test/" \
    #                  "19-03-29_01-35-14/checkpoints/run_00"

    # lr=1e-4, bs=1, [900, 900], GN
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-4_bs_1_GN/19-04-01_16-24-18/checkpoints/run_00"

    # lr=1e-4, bs=2, [900, 900], BN
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-4_bs_2_BN/19-04-02_10-58-55/checkpoints/run_00"

    # lr=1e-4, bs=2, [900, 900], GN
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-4_bs_2_GN/19-04-02_10-38-23/checkpoints/run_00"

    # lr=1e-4, bs=2, [900, 900], IN, broken
    # model_path = "/home/temp/moriz/checkpoints/retinanet/inbreast/new/" \
    #                  "inbreast_lr_1e-4_bs_2_IN/19-04-02_10-53-14/checkpoints/run_00"

    #--------------------------------------------------------------------------

    # DDSM, 900x900, n_train=all, n_val=all
    # model_path = "/home/temp/moriz/checkpoints/retinanet/ddsm/ddsm_crops_all/" \
    #                  "ddsm_mass_crops_all/" \
    #                  "19-04-09_10-49-41/checkpoints/run_00"

    # DDSM, 900x900, n_train=all, n_val=all, correctly pretrained
    model_path = "/home/temp/moriz/checkpoints/retinanet/ddsm/ddsm_crops_all/" \
                 "ddsm_mass_crops_all/" \
                 "19-04-13_16-53-33/checkpoints/run_00"

    # DDSM, IN, 900x900, n_train=all, n_val=all, correctly pretrained
    # model_path = "/home/temp/moriz/checkpoints/retinanet/ddsm/ddsm_crops/" \
    #              "ddsm_mass_crops_all_IN/" \
    #              "19-04-13_20-23-37/checkpoints/run_00"

    # flags and options
    settings = {"segment": False,
                "crop_size": [900, 900],
                "img_shape": None,
                "random_seed": 42,
                "norm": True,
                "type": "mass",
                "detection_only": True,
                "num_elements": None,
                "set": "val"}

    # _, test_paths, val_paths = inbreast_utils.load_single_set(inbreast_dir,
    #                                                  xls_file=xls_file,
    #                                                  train_size=0.7,
    #                                                  val_size=0.15,
    #                                                  type=settings["type"],
    #                                                  random_state=settings[
    #                                                      "random_seed"])
    #
    # # load INbreast data
    # dataset = LazyINbreastDataset(inbreast_dir,
    #                               inbreast_utils.load_pos_crops,
    #                               path_list = test_paths,
    #                               xls_file=xls_file,
    #                               **settings)


    _, _, val_paths = ddsm_utils.load_single_set(ddsm_dir,
                                                 csv_file=csv_mass_all,
                                                 train_size=0.7,
                                                 val_size=0.15,
                                                 random_state=42)

    dataset = LazyDDSMDataset(ddsm_dir,
                              ddsm_utils.load_pos_crops,
                              path_list= val_paths,
                              csv_file=csv_mass_all,
                              **settings)

    main(dataset,
         checkpoint_dir=model_path,
         start_epoch=5,
         end_epoch=75,
         step_size=5,
         plot=False,
         results_save_dir="/home/temp/moriz/validation/results/",
         **settings)