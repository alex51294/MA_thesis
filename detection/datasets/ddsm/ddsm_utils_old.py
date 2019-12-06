import pandas
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize

def load_DDSM_sample(img_path, csv_file, data_path):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # exception handling: if there are several ROIs, there are two or more
    # lines containing the same information
    label = data_frame.loc[img_path]['pathology']
    if not isinstance(label, str):
        tmp = label[:]
        label = tmp.iloc[0]

    # the label is still a string, but it must be numpy number
    # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
    if label == "MALIGNANT":
        label = np.asarray([1]).astype(np.float32)
    else:
        label = np.asarray([0]).astype(np.float32)

    # load image from path and load it as numpy array
    img_path = data_path + img_path
    img_sitk = sitk.ReadImage(img_path)
    img = sitk_image_to_data(img_sitk)
    img = img.transpose()

    # adjust the type
    img = img.astype(np.float32)

    # return a dict
    result_dict = {"data": img, "label": label}

    return result_dict

def load_DDSM_sample_v2(img_path, csv_file, data_path, img_shape,
                        n_channels=1):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # exception handling: if there are several ROIs, there are two or more
    # lines containing the same information
    label = data_frame.loc[img_path]['pathology']
    if not isinstance(label, str):
        tmp = label[:]
        label = tmp.iloc[0]

    # the label is still a string, but it must be numpy number
    # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
    if label == "MALIGNANT":
        label = np.asarray([1]).astype(np.float32)
    else:
        label = np.asarray([0]).astype(np.float32)

    # load image from path and load it as numpy array
    img_path = data_path + img_path
    img_sitk = sitk.ReadImage(img_path)
    img = sitk_image_to_data(img_sitk)

    # resize the image to the desired img_shape
    img = resize(img, img_shape, mode='reflect', anti_aliasing=True)
    img = np.reshape(img, (*img_shape, n_channels))
    img = img.transpose((len(img_shape), *range(len(img_shape))))

    # adjust the type
    img = img.astype(np.float32)

    # return a dict
    result_dict = {"data": img, "label": label}

    return result_dict

def load_DDSM_sample_v3(img_path, csv_file, data_path, img_shape,
                        n_channels=1):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    img_key = img_path

    # load image from path and load it as numpy array
    img_path = data_path + img_path
    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk)

    # # resize the image to the desired img_shape; resize requires the
    # # channels to be the last dimension, not the first one
    # img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    # img = resize(img, img_shape, mode='reflect', anti_aliasing=True)
    #
    # # reshape it to the desired amount of channels and transpose
    # img = np.reshape(img, (*img_shape, n_channels))
    # img = img.transpose((len(img_shape), *range(len(img_shape))))

    # adjust the type
    img = img.astype(np.float32)

    # built a dict for all information
    result_dict = {"data": img}

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the GLOBAL label (here the only one)
        label = data_frame.loc[img_key]['pathology']

        # the label is still a string, but it must be numpy number
        # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
        if label == "MALIGNANT":
            label = np.asarray([1]).astype(np.float32)
        else:
            label = np.asarray([0]).astype(np.float32)

        # add the label to the dictionary
        result_dict['label'] = label

        # extract the mask: this is more complicated
        # first, we select the folder that contains both, the ROI and the mask
        ROI_folder = data_frame.loc[img_key]['cropped image file path'] \
            .split("00000")[0]

        # get the file names in the folder
        [file_1, file_2] = os.listdir(data_path + ROI_folder)

        # find out which one is the actual ROI (and which the mask):
        # Read meta information (keys + values)
        reader = sitk.ImageFileReader()
        reader.SetFileName(data_path + ROI_folder + file_1)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        for k in reader.GetMetaDataKeys():
            if reader.GetMetaData(k) == "cropped images":
                ROI = file_1
                mask = file_2
                break
        else:
            ROI = file_2
            mask = file_1

        mask_path = data_path + ROI_folder + mask
        if mask_path[-1] == "\n":
            mask_path = mask_path[:-1]

        # mask image
        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)

        bbox_params = bounding_box(mask)

        # add bounding box parameter together with a label to the dictionary
        result_dict['mask_' + str(abnormality_id)] = [bbox_params, label]

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the GLOBAL label (here the only one)
            label = data_frame.loc[img_key]['pathology'].iloc[i]

            # the label is still a string, but it must be numpy number
            # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
            if label == "MALIGNANT":
                label = np.asarray([1]).astype(np.float32)
            else:
                label = np.asarray([0]).astype(np.float32)

            # add the label to the dictionary
            result_dict['label'] = label

            # extract the mask: this is more complicated
            # first, we select the folder that contains both, the ROI and the mask
            ROI_folder = data_frame.loc[img_key]['cropped image file path'] \
                    .iloc[i].split("00000")[0]

            # get the file names in the folder
            [file_1, file_2] = os.listdir(data_path + ROI_folder)

            # find out which one is the actual ROI (and which the mask):
            # Read meta information (keys + values)
            reader = sitk.ImageFileReader()
            reader.SetFileName(data_path + ROI_folder + file_1)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            for k in reader.GetMetaDataKeys():
                if reader.GetMetaData(k) == "cropped images":
                    ROI = file_1
                    mask = file_2
                    break
            else:
                ROI = file_2
                mask = file_1

            mask_path = data_path + ROI_folder + mask
            if mask_path[-1] == "\n":
                mask_path = mask_path[:-1]

            # mask image
            mask_sitk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_sitk)

            bbox_params = bounding_box(mask)


            abnormality_id_value = \
                data_frame.loc[img_key]['abnormality id'].iloc[i]

            # add bounding box parameter together with a label to the dictionary
            result_dict['mask_' + str(abnormality_id_value)] = [bbox_params, label]

    return result_dict


def load_DDSM_sample_v4(img_path, csv_file, data_path, img_shape=None,
                        n_channels=1):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # save the img_path as key for adressing in csv file
    img_key = img_path

    # load image from path and load it as numpy array
    img_path = data_path + img_path
    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk)

    # built a dict for all information
    result_dict = {"data": []}

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the GLOBAL label (here the only one)
        label = data_frame.loc[img_key]['pathology']

        # the label is still a string, but it must be numpy number
        # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
        if label == "MALIGNANT":
            label = np.asarray([1]).astype(np.float32)
        else:
            label = np.asarray([0]).astype(np.float32)

        # add the label to the dictionary
        result_dict['label'] = label

        # extract the mask: this is more complicated, since the csv files
        # are messed up, we would need to load BOTH files based on
        # their paths and look up in the meta information which one is which;
        # however, parts of the csv files are correct, so we assume that in
        # general the mask is saved under the path "ROI mask file path";
        # to be sure, we load the meta information and search for the key word
        # "cropped images" that indicates that this file is the cropped image
        # and not the mask; in this case we switch the paths and load the right
        # image

        # get the ROI and the mask paths
        ROI_path = data_path + \
                   data_frame.loc[img_key]['cropped image file path']
        mask_path = data_path + \
                    data_frame.loc[img_key]['ROI mask file path']

        # sometimes there is a distracting \n that must be removed
        if mask_path[-1] == "\n":
            mask_path = mask_path[:-1]

        if ROI_path[-1] == "\n":
            ROI_path = ROI_path[:-1]

        # find out which one is the actual ROI (and which the mask):
        # Read meta information (keys + values)
        reader = sitk.ImageFileReader()
        reader.SetFileName(mask_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        for k in reader.GetMetaDataKeys():
            if reader.GetMetaData(k) == "cropped images":
                mask_path = ROI_path
                break

        # mask image
        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)

        # get bbox parameter
        bbox_params = bounding_box(mask)

        # rescale the bbox parameter to fit with the new imaga size
        if img_shape is not None:
            scale_y = img_shape[1] / img.shape[1]
            scale_x = img_shape[0] / img.shape[2]
            bbox_params[0] = np.ceil(bbox_params[0] * scale_x)
            bbox_params[1] = np.ceil(bbox_params[1] * scale_y)
            bbox_params[2] = np.ceil(bbox_params[2] * scale_x)
            bbox_params[3] = np.ceil(bbox_params[3] * scale_y)

        # add bounding box parameter together with a label to the dictionary
        result_dict['mask_' + str(abnormality_id)] = [bbox_params, label]

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the GLOBAL label (here the only one)
            label = data_frame.loc[img_key]['pathology'].iloc[i]

            # the label is still a string, but it must be numpy number
            # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
            if label == "MALIGNANT":
                label = np.asarray([1]).astype(np.float32)
            else:
                label = np.asarray([0]).astype(np.float32)

            # add the label to the dictionary
            result_dict['label'] = label

            # same procedure as above
            # get the ROI and the mask paths
            ROI_path = data_path + \
                   data_frame.loc[img_key]['cropped image file path'].iloc[i]
            mask_path = data_path + \
                   data_frame.loc[img_key]['ROI mask file path'].iloc[i]

            # sometimes there is a distracting \n that must be removed
            if mask_path[-1] == "\n":
                mask_path = mask_path[:-1]

            if ROI_path[-1] == "\n":
                ROI_path = ROI_path[:-1]

                # find out which one is the actual ROI (and which the mask):
            # Read meta information (keys + values)
            reader = sitk.ImageFileReader()
            reader.SetFileName(mask_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            for k in reader.GetMetaDataKeys():
                if reader.GetMetaData(k) == "cropped images":
                    mask_path = ROI_path
                    break

            # mask image
            mask_sitk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_sitk)

            # get bbox parameter
            bbox_params = bounding_box(mask)

            # rescale the bbox parameter to fit with the new imaga size
            if img_shape is not None:
                scale_y = img_shape[1] / img.shape[1]
                scale_x = img_shape[0] / img.shape[2]
                bbox_params[0] = np.ceil(bbox_params[0] * scale_x)
                bbox_params[1] = np.ceil(bbox_params[1] * scale_y)
                bbox_params[2] = np.ceil(bbox_params[2] * scale_x)
                bbox_params[3] = np.ceil(bbox_params[3] * scale_y)


            abnormality_id_value = \
                data_frame.loc[img_key]['abnormality id'].iloc[i]

            # add bounding box parameter together with a label to the dictionary
            result_dict['mask_' + str(abnormality_id_value)] = [bbox_params, label]

    # resize the image to the desired img_shape;
    if img_shape is not None:
        result_dict['data'] = resize_image(img, img_shape, n_channels)
    else:
        result_dict['data'] = img

    # adjust the type
    img = img.astype(np.float32)

    return result_dict

# most actual load function
# changed saving format to fit other functionalities
# modification: from dict entries "mask_x": [mask, label] to ONE entry
# 'mask': [[mask_x, label_x], [mask_y, label_y], ...]
def load_DDSM_sample_v5(img_path, csv_file, data_path, img_shape=None,
                        n_channels=1):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # save the img_path as key for adressing in csv file
    img_key = img_path

    # load image from path and load it as numpy array
    img_path = data_path + img_path
    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk)

    # built a dict for all information
    result_dict = {"data": [], "label": None, "bbox": []}

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the GLOBAL label (here the only one)
        label = data_frame.loc[img_key]['pathology']

        # the label is still a string, but it must be numpy number
        # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
        if label == "MALIGNANT":
            label = np.asarray([1]).astype(np.float32)
        else:
            label = np.asarray([0]).astype(np.float32)

        # add the label to the dictionary
        result_dict['label'] = label

        # extract the mask: this is more complicated, since the csv files
        # are messed up, we would need to load BOTH files based on
        # their paths and look up in the meta information which one is which;
        # however, parts of the csv files are correct, so we assume that in
        # general the mask is saved under the path "ROI mask file path";
        # to be sure, we load the meta information and search for the key word
        # "cropped images" that indicates that this file is the cropped image
        # and not the mask; in this case we switch the paths and load the right
        # image

        # get the ROI and the mask paths
        ROI_path = data_path + \
                   data_frame.loc[img_key]['cropped image file path']
        mask_path = data_path + \
                    data_frame.loc[img_key]['ROI mask file path']

        # sometimes there is a distracting \n that must be removed
        if mask_path[-1] == "\n":
            mask_path = mask_path[:-1]

        if ROI_path[-1] == "\n":
            ROI_path = ROI_path[:-1]

        # find out which one is the actual ROI (and which the mask):
        # Read meta information (keys + values)
        reader = sitk.ImageFileReader()
        reader.SetFileName(mask_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        for k in reader.GetMetaDataKeys():
            if reader.GetMetaData(k) == "cropped images":
                mask_path = ROI_path
                break

        # mask image
        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)

        # get bbox parameter
        bbox_params = np.asarray(bounding_box(mask)).astype(np.float32)

        # rescale the bbox parameter to fit with the new imaga size
        if img_shape is not None:
            scale_y = img_shape[1] / img.shape[1]
            scale_x = img_shape[0] / img.shape[2]
            bbox_params[0] = np.ceil(bbox_params[0] * scale_x)
            bbox_params[1] = np.ceil(bbox_params[1] * scale_y)
            bbox_params[2] = np.ceil(bbox_params[2] * scale_x)
            bbox_params[3] = np.ceil(bbox_params[3] * scale_y)

        # add bounding box parameter together with a label to the dictionary
        result_dict['bbox'].append(np.append(bbox_params, label))

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the GLOBAL label (here the only one)
            label = data_frame.loc[img_key]['pathology'].iloc[i]

            # the label is still a string, but it must be numpy number
            # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
            if label == "MALIGNANT":
                label = np.asarray([1]).astype(np.float32)
            else:
                label = np.asarray([0]).astype(np.float32)

            # add the label to the dictionary
            result_dict['label'] = label

            # same procedure as above
            # get the ROI and the mask paths
            ROI_path = data_path + \
                   data_frame.loc[img_key]['cropped image file path'].iloc[i]
            mask_path = data_path + \
                   data_frame.loc[img_key]['ROI mask file path'].iloc[i]

            # sometimes there is a distracting \n that must be removed
            if mask_path[-1] == "\n":
                mask_path = mask_path[:-1]

            if ROI_path[-1] == "\n":
                ROI_path = ROI_path[:-1]

                # find out which one is the actual ROI (and which the mask):
            # Read meta information (keys + values)
            reader = sitk.ImageFileReader()
            reader.SetFileName(mask_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            for k in reader.GetMetaDataKeys():
                if reader.GetMetaData(k) == "cropped images":
                    mask_path = ROI_path
                    break

            # mask image
            mask_sitk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_sitk)

            # get bbox parameter
            bbox_params = np.asarray(bounding_box(mask)).astype(np.float32)

            # rescale the bbox parameter to fit with the new imaga size
            if img_shape is not None:
                scale_y = img_shape[1] / img.shape[1]
                scale_x = img_shape[0] / img.shape[2]
                bbox_params[0] = np.ceil(bbox_params[0] * scale_x)
                bbox_params[1] = np.ceil(bbox_params[1] * scale_y)
                bbox_params[2] = np.ceil(bbox_params[2] * scale_x)
                bbox_params[3] = np.ceil(bbox_params[3] * scale_y)


            abnormality_id_value = \
                data_frame.loc[img_key]['abnormality id'].iloc[i]

            # add bounding box parameter together with a label to the dictionary
            result_dict['bbox'].append(np.append(bbox_params, label))

    # convert the bbox to numpy array
    result_dict["bbox"] = np.asarray(result_dict["bbox"]).astype(np.float32)

    # resize the image to the desired img_shape;
    if img_shape is not None:
        result_dict['data'] = resize_image(img, img_shape, n_channels)
    else:
        result_dict['data'] = img

    # adjust the type
    result_dict['data'] = result_dict['data'].astype(np.float32)

    return result_dict

def resize_image(img, img_shape, n_channels):
    # resize requires the channels to be the last dimension, not the first one
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    img = resize(img, img_shape, mode='reflect', anti_aliasing=True)

    # reshape it to the desired amount of channels and transpose
    img = np.reshape(img, (*img_shape, n_channels))
    img = img.transpose((len(img_shape), *range(len(img_shape))))
    return img


def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.moveaxis(data, [0, 1, 2], [2, 0, 1])
    return data

def bounding_box(mask, margin=None):
    """Calculate bounding box coordinates of binary mask

    Parameters
    ----------
    mask : Image as Numpy array
        Binary mask
    margin : int, default: None
        margin to be added to min/max on each dimension

    Returns
    -------
    list
        bounding box center coordinates, width and height
        of the form (x_center, y_center, width, height)
    """
    # mask is in channel, y, x order
    # channels, ydim, xdim = mask.shape

    # determine where the mask values are not zero
    nz = np.where(mask[0,:,:] != 0)

    # get boundaries (minima and maxima)
    lower = [np.min(nz[0]), np.min(nz[1])]
    upper = [np.max(nz[0]), np.max(nz[1])]

    # add margin if specified
    if margin is not None:
        for axis in range(2):
            # make sure lower bound with margin is valid
            if lower[axis] - margin >= 0:
                lower[axis] -= margin
            else:
                lower[axis] = 0
            # make sure upper bound with margin is valid
            if upper[axis] + margin <= mask.shape[axis] - 1:
                upper[axis] += margin
            else:
                upper[axis] = mask.shape[axis] - 1

    # determine the total width and height of the mask
    height = upper[0] - lower[0]
    width = upper[1] - lower[1]

    # determine the center coordinates of the mask
    center_y = np.floor(lower[0] + height/2)
    center_x = np.floor(lower[1] + width/2)

    # return bbox parameters
    bbox = [center_x, center_y, width, height]
    return bbox


def load_DDSM_sample_v23(img_path, csv_file, data_path):
    # load image from path and load it as numpy array
    img_sitk = sitk.ReadImage(img_path)
    img = sitk_image_to_data(img_sitk)

    # built a dict for all information
    result_dict = {"data": img}

    # split img_path into pieces
    img_key = img_path.split(data_path)[1]

    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the label
        label = data_frame.loc[img_key]['pathology']
        result_dict['label'] = label

        # extract the ROI; however, this is slightly more complicated
        # first, we select the folder that contains both, the ROI and the mask
        ROI_folder = data_frame.loc[img_key]['cropped image file path'] \
                .split("00000")[0]

        # get the file names in the folder
        [file_1, file_2] = os.listdir(data_path + ROI_folder)

        # find out which one is the actual ROI (and which the mask):
        # Read meta information (keys + values)
        reader = sitk.ImageFileReader()
        reader.SetFileName(data_path + ROI_folder + file_1)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        for k in reader.GetMetaDataKeys():
            if reader.GetMetaData(k) == "cropped images":
                ROI = file_1
                mask = file_2
                break
        else:
            ROI = file_2
            mask = file_1

        ROI_path = data_path + ROI_folder + ROI
        if ROI_path[-1] == "\n":
            ROI_path = ROI_path[:-1]
        ROI_sitk = sitk.ReadImage(ROI_path)
        ROI = sitk_image_to_data(ROI_sitk)
        result_dict['ROI_' + str(abnormality_id)] = [ROI, label]
    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the label
            label = data_frame.loc[img_key]['pathology'].iloc[i]
            result_dict['label'] = label

            # extract the ROI; however, this is slightly more complicated
            # first, we select the folder that contains both, the ROI and
            # the mask
            ROI_folder = data_frame.loc[img_key]['cropped image file path'] \
                .iloc[i].split("00000")[0]

            # get the file names in the folder
            [file_1, file_2] = os.listdir(data_path + ROI_folder)

            # find out which one is the actual ROI (and which the mask):
            # Read meta information (keys + values)
            reader = sitk.ImageFileReader()
            reader.SetFileName(data_path + ROI_folder + file_1)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            for k in reader.GetMetaDataKeys():
                if reader.GetMetaData(k) == "cropped images":
                    ROI = file_1
                    mask = file_2
                    break
            else:
                ROI = file_2
                mask = file_1

            ROI_path = data_path + ROI_folder + ROI
            if ROI_path[-1] == "\n":
                ROI_path = ROI_path[:-1]
            ROI_sitk = sitk.ReadImage(ROI_path)
            ROI = sitk_image_to_data(ROI_sitk)

            abnormality_id_value = \
                data_frame.loc[img_key]['abnormality id'].iloc[i]

            result_dict['ROI_' + str(abnormality_id_value)] = [ROI, label]


    return result_dict

# to add: mask not binary (max value 255 not 1)
def get_mask_share(mask):
    mask_shape = mask.shape
    image_area = mask_shape[0]*mask_shape[1]
    mask_area = np.sum(mask)

    return mask_area/image_area

def print_meta_from_img(img_file):
    # Print out meta information (keys + values)
    reader = sitk.ImageFileReader()
    reader.SetFileName(img_file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        #if k == "0020|000d" or k == "0020|000e":
        print("({0}) = = \"{1}\"".format(k,v))

    print("Image Size: {0}".format(reader.GetSize()))
    print("Image PixelType: {0}"
          .format(sitk.GetPixelIDValueAsString(reader.GetPixelID())))


def plot_image(image, path=False):
    if path == True:
        img_sitk = sitk.ReadImage(image)
        img = sitk_image_to_data(img_sitk)
    else:
        img = image

    imgplot = plt.imshow(img[:, :, 0], cmap='Greys')
    plt.show()


def get_patient_list(csv_file):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file)

    patient_list = []

    for i in range(data_frame.shape[0]):
        patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
        if patient_id not in patient_list:
            patient_list.append(patient_id)

    return patient_list


def get_number_patients(csv_file):
    patient_list = get_patient_list(csv_file)

    return len(patient_list)


def get_patient_views(csv_file):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file)

    patient_view = {}

    for i in range(data_frame.shape[0]):
        patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
        if patient_id not in patient_view:
            patient_view[patient_id] = []

        view = data_frame.iloc[i]['left or right breast'] + "_" + \
                data_frame.iloc[i]['image view']
        if view not in patient_view[patient_id]:
            patient_view[patient_id].append(view)

    return patient_view

# one way of view distribution; a better one would be to consider the
# different possible constellations: all four, two for one side,
# one for one side
def get_view_distribution(csv_file):
    patient_list = get_patient_views(csv_file)
    view_dist = {'LEFT_MLO': 0, 'RIGHT_MLO': 0, 'LEFT_CC': 0, 'RIGHT_CC': 0}

    for i in patient_list:
        if 'LEFT_MLO' in patient_list[i]:
            view_dist['LEFT_MLO'] = view_dist['LEFT_MLO'] + 1
        if 'RIGHT_MLO' in patient_list[i]:
            view_dist['RIGHT_MLO'] = view_dist['RIGHT_MLO'] + 1
        if 'LEFT_CC' in patient_list[i]:
            view_dist['LEFT_CC'] = view_dist['LEFT_CC'] + 1
        if 'RIGHT_CC' in patient_list[i]:
            view_dist['RIGHT_CC'] = view_dist['RIGHT_CC'] + 1

    return view_dist


def get_number_mammograms(csv_file):
    view_dist = get_view_distribution(csv_file)

    return view_dist['LEFT_MLO'] + view_dist['RIGHT_MLO'] + \
                view_dist['LEFT_CC'] + view_dist['RIGHT_CC']


def get_label_distribution(csv_file):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file)
    label_dist = {"BENIGN":0, "MALIGNANT":0, "BENIGN_WITHOUT_CALLBACK":0}

    patient_list = []

    for i in range(data_frame.shape[0]):
        patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
        if patient_id in patient_list:
            continue
        patient_list.append(patient_id)
        label_dist[data_frame.iloc[i]['pathology']] = \
            label_dist[data_frame.iloc[i]['pathology']] + 1

    return label_dist

def get_assessment_distribution(csv_file):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file)
    assessment_dist = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0}

    patient_list = []

    for i in range(data_frame.shape[0]):
        patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
        if patient_id in patient_list:
            continue
        patient_list.append(patient_id)
        assessment_dist[str(data_frame.iloc[i]['assessment'])] = \
            assessment_dist[str(data_frame.iloc[i]['assessment'])] + 1

    return assessment_dist


def get_patient_laterality(csv_file):
    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file)

    patient_laterality = {}

    for i in range(data_frame.shape[0]):
        patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
        if patient_id not in patient_laterality:
            patient_laterality[patient_id] = []
        if data_frame.iloc[i]['left or right breast'] not in patient_laterality[patient_id]:
            patient_laterality[patient_id].append(data_frame.iloc[i]['left or right breast'])

    return patient_laterality


def get_laterality_distribution(csv_file):
    patient_list = get_patient_laterality(csv_file)
    laterality_dist = {"LEFT":0, "RIGHT":0, "BOTH":0 }

    for i in patient_list:
        if len(patient_list[i]) == 2:
            laterality_dist['BOTH'] = laterality_dist['BOTH'] + 1
        elif patient_list[i][0] == 'RIGHT':
            laterality_dist['RIGHT'] = laterality_dist['RIGHT'] + 1
        else:
            laterality_dist['LEFT'] = laterality_dist['LEFT'] + 1

    return laterality_dist

# def get_calc_type(csv_file):
#     # read csv file using pandas
#     data_frame = pandas.read_csv(csv_file)
#
#     patient_calc_type = {}
#
#     for i in range(data_frame.shape[0]):
#         patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
#         if patient_id not in patient_laterality:
#             patient_laterality[patient_id] = []
#         if data_frame.iloc[i]['left or right breast'] not in \
#                 patient_laterality[patient_id]:
#             patient_laterality[patient_id].append(
#                 data_frame.iloc[i]['left or right breast'])
#
#     return patient_laterality



def plot_barchart(x, y_calc_train, y_calc_test, y_mass_train,
                          y_mass_test, title, x_label, y_label, fig_nummer):

    fig = plt.figure(fig_nummer, figsize=(15,10))
    plt.subplot(221)
    plt.bar(x, y_calc_train, width=0.5)
    plt.ylim(0,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Calc Train")

    plt.subplot(222)
    plt.bar(x, y_calc_test, width=0.5)
    plt.ylim(0,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Calc Test")

    plt.subplot(223)
    plt.bar(x, y_mass_train, width=0.5)
    plt.ylim(0,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Mass Train")

    plt.subplot(224)
    plt.bar(x, y_mass_test, width=0.5)
    plt.ylim(0,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Mass Test")

    plt.suptitle(title)
    plt.show()


def plot_histogram(y_calc_train, y_calc_test, y_mass_train,
                          y_mass_test, title, x_label, y_label, fig_nummer):

    fig = plt.figure(fig_nummer, figsize=(15,10))
    plt.subplot(221)
    plt.hist(y_calc_train, bins='auto')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Calc Train")

    plt.subplot(222)
    plt.hist(y_calc_test, bins='auto')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Calc Test")

    plt.subplot(223)
    plt.hist(y_mass_train, bins='auto')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Mass Train")

    plt.subplot(224)
    plt.hist(y_mass_test, bins='auto')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Mass Test")

    plt.suptitle(title)
    plt.show()