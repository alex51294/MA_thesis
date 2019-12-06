import numpy as np
import random

from sklearn.model_selection import KFold

from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening, dilation


def create_crops(sample, crop_size=[600,600], overlap = 0.5, heatmap=False):
    """
        creates crop list out of data sample
    :param sample (dict): must be of the form {data, label, seg}
    :param crop_size (list/tuple): crop size that is desired (y,x)
    :param overlap (float): desired overlap in (y,x) direction;
            must be between 0 and 1

    :return crop_list (list): list with crops extracted from the image
    :return corner_list (list) : list containing the top left corner position
            of each crop relative to the original image ((x,y) format)
    """

    img_orig = sample["data"]
    seg_orig = sample["seg"]
    label_orig = sample["label"]

    # calculate the overlap dynamically so that it is at least as high as
    # the desired one
    num_crops_x = np.ceil(img_orig.shape[2] / crop_size[1])
    while True:
        overshoot_x = crop_size[1] * num_crops_x - img_orig.shape[2]
        overlap_x = (overshoot_x / (num_crops_x - 1))
        if overlap_x < overlap * crop_size[1]:
            num_crops_x += 1
            continue
        else:
            step_size_x = np.ceil(crop_size[1] - overlap_x).astype(np.int)
            break
    #print(overlap_x / crop_size[1])

    num_crops_y = np.ceil(img_orig.shape[1] / crop_size[0])
    while True:
        overshoot_y = crop_size[0] * num_crops_y - img_orig.shape[1]
        overlap_y = (overshoot_y / (num_crops_y - 1))
        if overlap_y < overlap * crop_size[0]:
            num_crops_y += 1
            continue
        else:
            step_size_y = np.ceil(crop_size[0] - overlap_y).astype(np.int)
            break
    #print(overlap_y / crop_size[0])

    # get the top left corner of each crop
    top_left_corner_x = np.arange(0,
                                  (step_size_x * num_crops_x).astype(np.int),
                                  step_size_x)

    top_left_corner_y = np.arange(0,
                                  (step_size_y * num_crops_y).astype(np.int),
                                  step_size_y)

    # correct the crops that go over the boundaries of the image
    for i in range(len(top_left_corner_x)):
        if (top_left_corner_x[i] + crop_size[1]) > img_orig.shape[2]:
            top_left_corner_x[i] = img_orig.shape[2] - crop_size[1]

    for i in range(len(top_left_corner_y)):
        if (top_left_corner_y[i] + crop_size[0]) > img_orig.shape[1]:
            top_left_corner_y[i] = img_orig.shape[1] - crop_size[0]

    # summarize in one list; this list is required to re-organize the crops
    # as one image
    corner_list = [(x, y) for x in top_left_corner_x for y in
                   top_left_corner_y]

    # cut out image parts based on the parameter and save them in the list
    image_list = [img_orig[:, y:y + crop_size[0], x:x + crop_size[1]] \
                  for x in top_left_corner_x for y in top_left_corner_y]

    # crop the segmentation the same way
    seg_list = [seg_orig[:, y:y + crop_size[0], x:x + crop_size[1]] \
                  for x in top_left_corner_x for y in top_left_corner_y]

    # create a new label_list that denotes the occurrence of lesions in
    # the single crops; for stability, discard lesions whose center is closer
    # than 25 pixels the closest boundary
    label_list = []
    for i in range(len(corner_list)):
        mask_area = np.sum(seg_list[i])
        if mask_area > 0:
            center = bounding_box(seg_list[i])[:,:2]
            dist = np.maximum(np.abs(crop_size[0]/2 - center[:, 0]),
                              np.abs(crop_size[1]/2 - center[:, 1]))

            if any(dist < (np.minimum(crop_size[0], crop_size[1])/2 - 25)):
                label_list.append(label_orig)
            else:
                label_list.append(-1)
        else:
            label_list.append(-1)

    # combine the image list, label list and bbox list
    crop_list = [{"data": image_list[j],
                  "seg": np.asarray(seg_list[j]),
                  "label": label_list[j]}
                 for j in range(len(image_list))]

    # if desired, return a "heatmap" that illustrates the subdivision into
    # crops; useful for WBC
    if heatmap:
        heatmap = np.zeros_like(img_orig[0, : ,:])
        for i in range(len(corner_list)):
            heatmap[corner_list[i][1]:corner_list[i][1] + crop_size[0],
            corner_list[i][0]:corner_list[i][0] + crop_size[1]] += 1
        return crop_list, corner_list, heatmap

    return crop_list, corner_list

def create_channel_info(image, crop_size, num_channels,
                        return_num_crops=False, mode="crops"):
    if mode == "crops":
        # calculate the overlap dynamically so that it is at least as high as
        # the desired one
        num_crops_y = np.ceil(image.shape[1] / crop_size[0])
        num_crops_x = np.ceil(image.shape[2] / crop_size[1])

        overshoot_x = crop_size[1] * num_crops_x - image.shape[2]
        overlap_x = (overshoot_x / (num_crops_x - 1))
        step_size_x = np.ceil(crop_size[1] - overlap_x).astype(np.int)

        overshoot_y = crop_size[0] * num_crops_y - image.shape[1]
        overlap_y = (overshoot_y / (num_crops_y - 1))
        step_size_y = np.ceil(crop_size[0] - overlap_y).astype(np.int)

        # get the top left corner of each crop
        top_left_corner_x = np.arange(0,
                                      (step_size_x * num_crops_x).astype(np.int),
                                      step_size_x)

        top_left_corner_y = np.arange(0,
                                      (step_size_y * num_crops_y).astype(np.int),
                                      step_size_y)

        # correct the crops that go over the boundaries of the image
        for i in range(len(top_left_corner_x)):
            if (top_left_corner_x[i] + crop_size[1]) > image.shape[2]:
                top_left_corner_x[i] = image.shape[2] - crop_size[1]

        for i in range(len(top_left_corner_y)):
            if (top_left_corner_y[i] + crop_size[0]) > image.shape[1]:
                top_left_corner_y[i] = image.shape[1] - crop_size[0]

        # cut out image parts based on the parameter and save them in the list
        crop_list = [image[:, y:y + crop_size[0], x:x + crop_size[1]] \
                      for x in top_left_corner_x for y in top_left_corner_y]

        true_num_crops = len(crop_list)

        if true_num_crops != num_channels:
            tissue_amount = { "%d" %i : np.sum(crop_list[i]) for i in range(len(crop_list))}

            if len(crop_list) > num_channels:
                reduced_crop_list = []
                for key, value in sorted(tissue_amount.items(),
                                         key=lambda item: item[1], reverse=True):
                    reduced_crop_list.append(crop_list[int(key)])

                    if len(reduced_crop_list) == num_channels:
                        break

                crop_list = reduced_crop_list
            else:
                enlarged_crop_list = crop_list
                for key, value in sorted(tissue_amount.items(),
                                         key=lambda item: item[1], reverse=True):
                    enlarged_crop_list.append(crop_list[int(key)])

                    if len(enlarged_crop_list) == num_channels:
                        break
                crop_list = enlarged_crop_list

    elif mode == "sampled":
        step_size_y = np.ceil(image.shape[1] / crop_size[0]).astype(np.int32)
        step_size_x = np.ceil(image.shape[2] / crop_size[1]).astype(np.int32)

        # cut out image parts based on the parameter and save them in the list
        crop_list = []
        for i in range(np.minimum(step_size_y, step_size_x)):
            crop = image[:, i:image.shape[1]:step_size_y, i:image.shape[2]:step_size_x]
            crop_list.append(crop)

        true_num_crops = len(crop_list)
        if true_num_crops > num_channels:
            crop_list = crop_list[:num_channels]
    else:
        raise ValueError("Unknown mode!")

    if not return_num_crops:
        return crop_list
    else:
        return crop_list, true_num_crops

def resize_image(img, img_shape):
    """Resize image to the desired size

        Parameters
        ----------
        img : Image as Numpy array
            image to resize
        img_shape : tuple
            tuple: img_shape that is desired (height,width)

        Returns
        -------
        list
            bounding box center coordinates, width and height
            of the form (x_center, y_center, width, height)
        """
    # resize requires the channels to be the last dimension, not the first one
    if np.asarray(img.shape).min() != img.shape[2]:
        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])

    # check for type of img_shape
    if isinstance(img_shape, int):
        pos_short_side = np.asarray(img.shape[0:2]).argmin()
        pos_long_side = np.asarray(img.shape[0:2]).argmax()
        short_side = img_shape
        ratio = float(img.shape[pos_long_side]/img.shape[pos_short_side])
        long_side = int(img_shape * ratio)
        img_shape = (long_side, short_side)

    # if not isinstance(img.dtype, int):
    #     img = img.astype(np.int32)

    # added preserve_range, not sure if correct
    img = resize(img, img_shape, mode='reflect', anti_aliasing=False,
                 preserve_range=True)

    # reshape it to the desired amount of channels and transpose
    img = np.reshape(img, (*img_shape, img.shape[2]))
    img = img.transpose((len(img_shape), *range(len(img_shape))))

    return img

def bounding_box(mask, margin=None):
    """Calculate bounding box coordinates of binary mask

    Parameters
    ----------
    mask : Image as Numpy array
        Binary mask
    offset : float, default: None
        percentage of width/hight to be added up

    Returns
    -------
    list
        bounding box center coordinates, width and height
        of the form (x_center, y_center, width, height)
    """
    # the mask might be in (c, y, x) order
    if len(mask.shape) > 2:
        mask = mask[0]

    # if there is no mask present, return an empty bbox (point)
    # TODO: desiced whether this approach is appropriate
    if np.sum(mask) == 0:
        return np.array([[0, 0, 1, 1]]).astype(np.float32)

    # label the single masks
    mask_labeled, num = label(mask, return_num=True)

    # get bbox parameter
    bbox_params = []
    for i in range(1, num+1):
        # select one labeled lesion per iteration
        mask_i = mask_labeled == i
        #mask_i = mask_i.reshape((1, mask_i.shape[0], mask_i.shape[1]))

        # determine where the mask values are not zero
        nz = np.where(mask_i != 0)

        # get boundaries (minima and maxima)
        lower = [np.min(nz[0]), np.min(nz[1])]
        upper = [np.max(nz[0]), np.max(nz[1])]

        # determine the total width and height of the mask
        height = upper[0] - lower[0]
        width = upper[1] - lower[1]

        if margin is not None:
            if margin < 0 or margin > 1:
                ValueError("Margin must be a value in [0,1]!")
            offset = [np.floor(height * margin), np.floor(width * margin)]

            # make sure bounds are valid
            for axis in range(2):

                if lower[axis] - offset[axis] >= 0:
                    lower[axis] -= offset[axis]
                else:
                    lower[axis] = 0

                if upper[axis] + offset[axis] <= mask.shape[axis] - 1:
                    upper[axis] += offset[axis]
                else:
                    upper[axis] = mask.shape[axis] - 1

            height = upper[0] - lower[0]
            width = upper[1] - lower[1]

        # save (for now) in [x_1,y_1, x_2, x_2] format
        bbox_params.append([lower[1], lower[0], upper[1], upper[0]])


    # if used on cropped masks, small overlapping bbox might appear
    # that should be removed
    # reference = bbox_params[0]
    # num_bbox_params = len(bbox_params)
    # for i in range(num_bbox_params):
    #     area = (bbox_params[i][3] - bbox_params[i][1]) * (bbox_params[i][2] - bbox_params[i][0])
    #     if area > (reference[3] - reference[1]) * (reference[2] - reference[0]):
    #         reference = bbox_params[i]
    #
    # i = 0
    # while i < num_bbox_params:
    #     if (bbox_params[i][0] > reference[0] and \
    #         bbox_params[i][1] > reference[1]) or \
    #         (bbox_params[i][2] < reference[2] and \
    #         bbox_params[i][3] < reference[3]):
    #         bbox_params.pop(i)
    #         num_bbox_params -= 1
    #         i = 0
    #         continue
    #     i += 1

    if len(bbox_params) == 0:
        return np.array([[0, 0, 1, 1]]).astype(np.float32)

    # convert into numpy and save in [x_center, y_center, width ,hight] format
    bbox_params = np.asarray(bbox_params).astype(np.float32)
    size = bbox_params[:,2:] - bbox_params[:,0:2]
    bbox_params[:,0:2] = np.floor(bbox_params[:,0:2] + size/2)
    bbox_params[:,2:] = size

    # return bbox parameters
    return bbox_params

def segment_breast(sample):
    """
    Segments the breast and cuts the backround; if provided, the bboxes are
        aligned
    :param sample (dict): Sample of the form {data, label, bbox, seg}

    :return: sample (dict)
    """

    # segment the breast based on the otsu threshold
    otsu_thr = threshold_otsu(sample["data"])
    otsu_mask = sample["data"] > otsu_thr

    # preprocess mask to suppress single outliners
    otsu_mask = closing(otsu_mask[0], square(3))

    # label the single regions occuring in the mask
    mask_labeled, num = label(otsu_mask, return_num=True)

    # start_time = time()
    # for i in range(1, num+1):
    #     # select one labeled lesion per iteration
    #     mask_i = np.int32(mask_labeled == i)
    #     props = regionprops(mask_i)
    #
    #     if props[0].area > max_area:
    #         max_area = props[0].area
    #         otsu_bbox_list.append(props[0].bbox)
    # print(time() - start_time)

    # sort out artifacts (first step)
    otsu_bbox_list = []
    max_area = 0
    for region in regionprops(mask_labeled):
        if region.area > max_area:
            max_area = region.area
            otsu_bbox_list.append(region.bbox)

    if len(otsu_bbox_list) == 0:
        otsu_bbox_list.append(region.bbox)

    # sometimes, artifacts might still be present; take the largest bbox as the
    # one representing the breast (second step, if required)
    if len(otsu_bbox_list) > 1:
        max_area = 0
        for i in range(len(otsu_bbox_list)):
            area = (otsu_bbox_list[i][3] - otsu_bbox_list[i][1]) * \
                   (otsu_bbox_list[i][2] - otsu_bbox_list[i][0])
            if area > max_area:
                max_area = area
                otsu_bbox = np.asarray(otsu_bbox_list[i])
    else:
        otsu_bbox = np.asarray(otsu_bbox_list[0])

    # crop the image
    sample["data"] = sample["data"][:, otsu_bbox[0]: otsu_bbox[2],
               otsu_bbox[1]:otsu_bbox[3]]

    # crop segmentation mask (if provided)
    if "seg" in sample.keys():
        sample["seg"] = sample["seg"][:, otsu_bbox[0]: otsu_bbox[2],
               otsu_bbox[1]:otsu_bbox[3]]

    # second filter stage: histogram based
    #sample = segment_breast_via_hist(sample)

    return sample

def segment_breast_via_hist(sample, thr_x=0.05, thr_y=0.05):

    hist_y = np.sum(sample["data"][0], axis=0)
    hist_y = hist_y / np.max(hist_y)
    x_range = np.argwhere(hist_y > np.min(hist_y) + thr_x)
    x_lower_limit = x_range[0, 0]
    x_upper_limit = x_range[-1, 0]

    hist_x = np.sum(sample["data"][0], axis=1)
    hist_x = hist_x / np.max(hist_x)
    y_range = np.argwhere(hist_x > np.min(hist_x) + thr_y)
    y_lower_limit = y_range[0, 0]
    y_upper_limit = y_range[-1, 0]

    sample["data"] = sample["data"][:, y_lower_limit:y_upper_limit, x_lower_limit:x_upper_limit]
    sample["seg"] = sample["seg"][:, y_lower_limit:y_upper_limit, x_lower_limit:x_upper_limit]

    return sample

# my old segmentation script
def segment_breast_old(sample):
    """
    Segments the breast and cuts the backround; if provided, the bboxes are
        aligned
    :param sample (dict): Sample of the form {data, label, bbox, seg}

    :return: sample (dict)
    """

    # segment the breast and cut the image
    otsu_thr = threshold_otsu(sample["data"])
    otsu_mask = sample["data"] > otsu_thr
    otsu_bbox_list = bounding_box(otsu_mask)

    # sometimes, artifacts might be present; take the largest bbox as the one
    # representing the breast
    max_area = 0
    for i in range(len(otsu_bbox_list)):
        area = otsu_bbox_list[i][2] * otsu_bbox_list[i][3]
        if area > max_area:
            max_area = area
            otsu_bbox = otsu_bbox_list[i]

    # get the left top and bottom right corner of the otsu bbox
    otsu_corners = np.array([otsu_bbox[0] - np.floor(otsu_bbox[2] / 2),
                             otsu_bbox[0] + np.floor(otsu_bbox[2] / 2),
                             otsu_bbox[1] - np.floor(otsu_bbox[3] / 2),
                             otsu_bbox[1] + np.floor(otsu_bbox[3] / 2)]).astype(int)

    # crop the image
    sample["data"] = sample["data"][:, otsu_corners[2]: otsu_corners[3],
               otsu_corners[0]:otsu_corners[1]]

    # crop segmentation mask (if provided)
    if "seg" in sample.keys():
        sample["seg"] = sample["seg"][:, otsu_corners[2]: otsu_corners[3],
               otsu_corners[0]:otsu_corners[1]]

    return sample

def get_patient_dict(paths, dataset_type="INbreast"):
    # dict for separating based on patient id
    patient_dict = {}

    # iterate over the paths and summarize different views and sides of the
    # patient to one entry
    for i in range(len(paths)):
        if dataset_type == "INbreast":
            patient_id = paths[i].split("/")[-1].split("_")[1]
        elif dataset_type == "DDSM":
            patient_id = paths[i].split("/")[-4].split("_")[2]
        else:
            raise ValueError("Unknown dataset type!")

        if patient_id in patient_dict:
            patient_dict[patient_id].append(paths[i])
        else:
            patient_dict[patient_id] = [paths[i]]

    return patient_dict

def split_paths_patientwise(paths, dataset_type, train_size=0.7,
                            val_size=None, random_state=None):
    # get patient dict
    patient_dict = get_patient_dict(paths, dataset_type)

    # if a random seed is given, shuffle the data patient wise
    if random_state:
        keys = list(patient_dict.keys())
        random.seed(random_state)
        random.shuffle(keys)
        patient_dict = {key: patient_dict[key] for key in keys}

    # seperate into distinct train, val and test sets
    train, val, test = [], [], []
    patient_keys = list(patient_dict.keys())
    for i in range(len(patient_dict)):
        if len(train) < np.floor(train_size * len(paths)):
            train += patient_dict[patient_keys[i]]
        elif val_size is not None and len(val) < np.floor(val_size * len(paths)):
            val += patient_dict[patient_keys[i]]
        else:
            test += patient_dict[patient_keys[i]]

    return train, test, val


def kfold_patientwise(paths, dataset_type, num_splits=5, shuffle=True,
                      random_state=None):
    # get patient dict
    patient_dict = get_patient_dict(paths, dataset_type)

    patient_ids = np.asarray(list(patient_dict.keys()))

    fold = KFold(n_splits=num_splits,
                 shuffle=shuffle,
                 random_state=random_state)

    train_splits = []
    test_splits = []

    for train_idx, test_idx in fold.split(patient_ids):
        train_paths = [path for key in patient_ids[train_idx] for path in
                      patient_dict[key]]
        train_splits.append(train_paths)

        test_paths = [path for key in patient_ids[test_idx] for path in
                      patient_dict[key]]
        # train_splits.append(patient_ids[train_idx])
        test_splits.append(test_paths)

    return train_splits, test_splits

