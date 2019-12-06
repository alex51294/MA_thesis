import torch
from tqdm import tqdm
import numpy as np
import os
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from detection.datasets import LazyINbreastDataset
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors

from scipy.stats import norm

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

    if "fold" in settings and settings["fold"] is not None:
        fold = "0%d" % settings["fold"]
    else:
        fold = "00"

    # set WBC factors
    crop_center = np.asarray(crop_size) / 2.0
    norm_pdf_var =  np.int32(min(crop_size[0], crop_size[1]) / 2. - 50)

    # create directory and file name
    cp_dir_date = checkpoint_dir.split("/")[-3]
    results_save_dir = results_save_dir + str(cp_dir_date) + "/" + "fold_" + \
                       fold + "_" + set +  "/image_level_" +  \
                       str(start_epoch) + "_" + str(end_epoch) + "_" + str(step_size)

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
        checkpoint_path = checkpoint_dir + "/checkpoint_epoch_" + str(epoch) + ".pth"

        # load model
        if device == "cpu":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path)
        model = RetinaNet(**checkpoint['init_kwargs']).eval()
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.to(device)


        model_results_dict = {}

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                torch.cuda.empty_cache()

                # get image data
                test_data = dataset[i]

                # generate bboxes
                gt_bbox = utils.bounding_box(test_data["seg"])
                gt_bbox = torch.Tensor(gt_bbox).to(device)

                # generate crops
                crop_list, corner_list, heatmap = utils.create_crops(test_data,
                                                            crop_size=crop_size,
                                                            heatmap=True)

                # define list for predicted bboxes in crops
                image_bboxes = []
                image_scores = []
                crop_center_factor = []
                heatmap_factor = []

                # iterate over crops
                for j in range(0, len(crop_list)):
                    #CROP LEVEL
                    torch.cuda.empty_cache()
                    test_image = torch.Tensor(crop_list[j]['data']).to(device)

                    # predict anchors and labels for the crops using the loaded model
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

                model_results_dict["image_%d" %i] = {"gt_list": gt_bbox,
                                             "box_list": image_bboxes,
                                             "score_list": image_scores,
                                             "merging_utils": {"ccf": crop_center_factor,
                                                               "hf": heatmap_factor}}

                # # convert GT bbox to tensor
                # gt_bbox = torch.Tensor(gt_bbox).to(device)
                #
                # if len(crop_bbox) > 0:
                #     if merging_method == "NMS":
                #         # merge overlapping bounding boxes using NMS
                #         image_bbox, score_bbox = eval_utils.nms(crop_bbox,
                #                                                 score_bbox,
                #                                                 0.2)
                #     elif merging_method == "WBC":
                #         # merge overlapping bounding boxes using WBC
                #         #image_bbox, score_bbox = eval_utils.wbc(crop_bbox, score_bbox, 0.2)
                #
                #         # merge overlapping bounding boxes using my merging
                #         image_bbox, score_bbox = \
                #             eval_utils.my_merging(crop_bbox,
                #                                   score_bbox,
                #                                   crop_center_factor,
                #                                   heatmap_factor,
                #                                   thr=0.2)
                #     else:
                #         raise KeyError("Merging method is not supported.")
                #
                #
                #     # calculate the required rates for the FROC metric
                #     tp_crop, fp_crop, fn_crop = \
                #         eval_utils.calc_tp_fn_fp(gt_bbox,
                #                                  image_bbox,
                #                                  score_bbox,
                #                                  confidence_values=confidence_values)
                #     tp_list.append(tp_crop)
                #     fp_list.append(fp_crop)
                #     fn_list.append(fn_crop)
                #
                #     # determine the overlap of detected bbox with the ground truth
                #     box_label = eval_utils.gt_overlap(gt_bbox, image_bbox)
                #     box_label_list.append(box_label)
                #     box_score_list.append(score_bbox)
                #
                # else:
                #     tp_list.append([torch.tensor(0, device=device) for tp
                #                     in range(len(confidence_values))])
                #     fp_list.append([torch.tensor(0, device=device) for fp
                #                     in range(len(confidence_values))])
                #     fn_list.append([torch.tensor(1, device=device) for fn
                #                     in range(len(confidence_values))])

        # DATASET LEVEL
        total_results_dict[str(epoch)] = model_results_dict

    # MODELS LEVEL
    with open(results_save_dir + "/results", "wb") as result_file:
        torch.save(total_results_dict, result_file)


if __name__ == '__main__':
    # path to data directory
    inbreast_dir = '/images/Mammography/INbreast/AllDICOMs/'

    # paths to csv files containing labels (and other information)
    xls_file = '/images/Mammography/INbreast/INbreast.xls'

    # path to image save directory
    image_save_dir = "/home/temp/moriz/validation/"

    # Adam training with bs=4, lr=1-e4, e_max = 500, s=42,
    # crop_size = [900, 900], unsegmented
    checkpoint_dir = "/home/temp/moriz/checkpoints/retinanet/inbreast/" \
                     "inbreast_CV_lr_1e-4_b_4_e_300_cs_900_900_unsegmented/" \
                     "19-02-27_17-56-21/checkpoints/"



    paths = inbreast_utils.get_paths(inbreast_dir, xls_file=xls_file)

    train_splits, _ = utils.kfold_patientwise(paths,
                                              dataset_type="INbreast",
                                              num_splits=5,
                                              shuffle=True,
                                              random_state=42)

    for i in range(4,5):
        # flags and options
        settings = {"segment": False,
                    "crop_size": [900, 900],
                    "img_shape": None,
                    "fold": i,
                    "set": "val"}

        _, val_paths, _ = \
            utils.split_paths_patientwise(train_splits[i],
                                          dataset_type="INbreast",
                                          train_size=0.9)

        # load INbreast data
        dataset = LazyINbreastDataset(inbreast_dir,
                                      inbreast_utils.load_sample,
                                      path_list=val_paths,
                                      xls_file=xls_file,
                                      img_shape=settings["img_shape"],
                                      num_elements=None,
                                      segment=settings["segment"],
                                      offset=None)



        main(dataset,
             checkpoint_dir = checkpoint_dir + "run_0%d" %i,
             start_epoch=5,
             end_epoch=200,
             step_size=5,
             results_save_dir="/home/temp/moriz/validation/results/",
             **settings)