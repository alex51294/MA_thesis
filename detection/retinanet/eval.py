import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np

import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors
from detection.retinanet.retinanet_utils import box_iou, change_box_order

def eval(dataset, model_path, plot=False):
    # device
    device = 'cuda'

    # load model
    checkpoint = torch.load(model_path)
    model = RetinaNet(**checkpoint['init_kwargs']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # hyperparams
    crop_size = [600, 600]
    overlapped_boxes = 0.5
    confidence_values = np.arange(0.5,1,0.05)
    tpr_list = []
    fppi_list = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            torch.cuda.empty_cache()

            # get image data
            test_data = dataset[i]


            # crop background
            test_data = inbreast_utils.segment_breast(test_data)
            image_bbox = utils.bounding_box(dataset[i]["seg"])

            # generate crops
            crop_list, corner_list = inbreast_utils.create_crops(test_data)

            # define list for predicted bboxes in crops
            crop_bbox = []
            score_bbox = []

            # plot the image with the according bboxes
            if plot:
                # plot image
                plt.figure(1, figsize=(15, 10))
                fig, ax = plt.subplots(1)

                ax.imshow(test_data["data"][0, :, :], cmap='Greys_r')

                # show bboxes as saved in data (in red with center)
                for l in range(len(image_bbox)):
                    pos = tuple(image_bbox[l][0:2])
                    plt.plot(pos[0], pos[1], 'r.')
                    width = image_bbox[l][2]
                    height = image_bbox[l][3]
                    pos = (pos[0] - np.floor(width / 2),
                           pos[1] - np.floor(height / 2))

                    # Create a Rectangle patch
                    rect = patches.Rectangle(pos, width, height, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

            # iterate over crops
            for j in tqdm(range(0, len(crop_list))):
                torch.cuda.empty_cache()
                test_image = torch.Tensor(crop_list[j]['data']).to(device)
                test_bbox = utils.bounding_box(crop_list[j]['seg'])

                # predict anchors and labels for the crops using the loaded model
                anchor_preds, cls_preds = model(test_image.unsqueeze(0))

                # convert the predicted anchors to bboxes
                anchors = Anchors()
                boxes, labels, score = anchors.generateBoxesFromAnchors(
                    anchor_preds[0].to('cpu'),
                    cls_preds[0].to('cpu'),
                    tuple(test_image.shape[1:]),
                    cls_tresh=0.05)

                # correct the predicted bboxes
                for k in range(len(boxes)):
                    center_corrected = boxes[k][0:2] + \
                                       torch.Tensor(corner_list[j])
                    crop_bbox.append(torch.cat((center_corrected,
                                               boxes[k][2:])))
                    score_bbox.append(score[k])

            # merge overlapping bounding boxes
            crop_bbox, score_bbox = merge(crop_bbox, score_bbox)

            # calculate the FROC metric (TPR vs. FPPI)
            tpr_int = []
            fppi_int = []
            image_bbox = change_box_order(torch.Tensor(image_bbox),
                                         order='xywh2xyxy').to('cpu')
            iou_thr = 0.2
            for j in confidence_values:
                current_bbox = crop_bbox[score_bbox > j]

                if len(current_bbox) == 0:
                    tpr_int.append(torch.Tensor([0]))
                    fppi_int.append(torch.Tensor([0]))
                    continue
                    #break

                iou_matrix = box_iou(image_bbox,
                                     change_box_order(current_bbox,
                                                      order="xywh2xyxy"))
                iou_matrix = iou_matrix > iou_thr

                # true positives are the lesions that are recognized
                tp = iou_matrix.sum()

                # false negatives are the lesions that are NOT recognized
                fn = image_bbox.shape[0] - tp

                # true positive rate
                tpr = tp.type(torch.float32) / (tp + fn).type(torch.float32)
                tpr = torch.clamp(tpr, 0, 1)

                # number of false positives per image
                fp = (current_bbox.shape[0] - tp).type(torch.float32)

                tpr_int.append(tpr)
                fppi_int.append(fp)
            tpr_list.append(tpr_int)
            fppi_list.append(fppi_int)


            if plot:
                # show the predicted bboxes (in blue)
                print("Number of detected bboxes: {0}".format(len(crop_bbox)))
                keep = score_bbox > 0.5
                crop_bbox = crop_bbox[keep]
                score_bbox = score_bbox[keep]
                for j in range(len(crop_bbox)):
                    width = crop_bbox[j][2]
                    height = crop_bbox[j][3]
                    pos = (crop_bbox[j][0] - torch.floor(width / 2),
                           crop_bbox[j][1] - torch.floor(height / 2))

                    # Create a Rectangle patch
                    rect = patches.Rectangle(pos, width, height, linewidth=1,
                                             edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    ax.annotate("{:.2f}".format(score_bbox[j]), pos,
                                fontsize=6,
                                xytext=(pos[0]+10, pos[1]-10))

                    print("BBox params: {0}, score: {1}".format(crop_bbox[j],
                                                                score_bbox[j]))
                plt.show()

            #     fig.savefig("../plots/" + "_".join(model_path.split("/")[5:8]) + ".png")

        # calculate FROC over all test images
        tpr_list = np.asarray(tpr_list)
        tpr = np.sum(tpr_list, axis=0) / tpr_list.shape[0]

        fppi_list = np.asarray(fppi_list)
        fppi = np.sum(fppi_list, axis=0) / fppi_list.shape[0]

    # plt.figure(1)
    # plt.ylim(0, 1.1)
    # plt.xlabel("False Positve per Image (FPPI)")
    # plt.ylabel("True Positive Rate (TPR)")
    # plt.title("Free Response Operating Characteristic (FROC)")
    # plt.plot(np.asarray(fppi), np.asarray(tpr), "rx-")
    # plt.show()

    return tpr, fppi


def merge(bboxes, score):
    # merge overlapping bounding boxes

    # bboxes and score must be tensors
    if isinstance(bboxes, list):
        bboxes = torch.stack(bboxes)

    if isinstance(score, list):
        score = torch.stack(score)

    # sort the score in descending order, adjust bboxes accordingly
    score, indices = torch.sort(score, descending=True)
    bboxes = bboxes[indices]

    # limit the amount of bboxes to 300 (or less)
    if len(score) > 300:
        limit = 300
    else:
        limit = len(score)
    score = score[0:limit]
    bboxes = bboxes[0:limit]

    # choose the highest scoring bboxes as starting point; add any bbox that is
    # not yet in the list and has no IoU value higher than 0.2 with every other
    # bbox in the list
    result_bbox = [bboxes[0]]
    result_score = [score[0]]
    for j in range(1, len(bboxes)):
        iou = box_iou(torch.stack(result_bbox),
                      bboxes[j].view(1, -1),
                      "xywh")
        if any(iou > 0.2):
            continue
        else:
            result_bbox.append(bboxes[j])
            result_score.append(score[j])


    bboxes = torch.stack(result_bbox)
    score = torch.stack(result_score)

    return bboxes, score