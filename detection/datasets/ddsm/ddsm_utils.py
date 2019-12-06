import pandas
import SimpleITK as sitk
import numpy as np
from detection.datasets import utils
from tqdm import tqdm
from skimage.morphology import closing, square, opening, dilation
import os

def load_sample(img_path, data_path, **kwargs):
    if "type" in kwargs:
        type = kwargs["type"]
    else:
        type = "mass"

    if "csv_file" in kwargs:
        csv_file = kwargs["csv_file"]
    elif type == "mass":
        csv_file = '/home/temp/moriz/data/mass_case_description_train_set.csv'
    else:
        csv_file = '/home/temp/moriz/data/calc_case_description_train_set.csv'

    # built a dict for all information
    result_dict = {}

    # save the img_path as key for addressing in csv file
    img_key = img_path.split(data_path)[1]

    # load image from path and load it as numpy array
    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk)

    # load mask
    mask = get_labeled_mask(csv_file, img_key, data_path)
    mask[0, :, :] = refine_mask(mask[0, :, :], type)

    if "img_shape" in kwargs and kwargs['img_shape'] is not None:
        img_shape = kwargs['img_shape']

        if len(img_shape) == 1:
            # determine which is the short side
            if img.shape[1] < img.shape[2]:
                pos_short_side = 1
                pos_long_side = 2
            else:
                pos_short_side = 2
                pos_long_side = 1

            # calculate the aspect ratio to scale the remaining side
            # and save the new shape
            ratio = float(img.shape[pos_long_side] / img.shape[pos_short_side])
            img_shape = (int(img_shape[0] * ratio), img_shape[0])
        else:
            img_shape = tuple(img_shape)
    else:
        img_shape = None

    # normalize to zero mean and unit variance (if desired)
    if "norm" in kwargs and kwargs["norm"]:
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            img[i, :, :] = (img[i, :, :] - np.mean(img[i, :, :])) / np.std(
                img[i, :, :])

    # if desired, crop the background and segment the breast
    if "segment" in kwargs and kwargs["segment"]:
        # sample = utils.segment_breast({"data": img, "seg": mask})
        sample = utils.segment_breast_via_hist({"data": img, "seg": mask})
        img = sample["data"]
        mask = sample["seg"]

    if "shape_limit" in kwargs and kwargs["shape_limit"] is not None:
        shape_limit = kwargs["shape_limit"]
        scale_factor = min(shape_limit[0] / img.shape[1],
                           shape_limit[1] / img.shape[2])

        img_shape = (min(shape_limit[0], int(np.ceil(img.shape[1]*scale_factor))),
                     min(shape_limit[1], int(np.ceil(img.shape[2]*scale_factor))))

        # if kwargs["shape_limit"] == "1080Ti":
        #     scale_factor = np.sqrt(3.4e6 / (img.shape[1] * img.shape[2]))
        #
        #     img_shape = (int(img.shape[1] * scale_factor),
        #                  int(img.shape[2] * scale_factor))
        # else:
        #     scale_factor = np.sqrt(1.7e6 / (img.shape[1] * img.shape[2]))
        #
        #     img_shape = (int(img.shape[1] * scale_factor),
        #                  int(img.shape[2] * scale_factor))
    else:
        shape_limit = None

    # resize image (if desired)
    if img_shape is not None or shape_limit is not None:
        img = utils.resize_image(img, img_shape)
        mask = utils.resize_image(mask, img_shape)

    # save all values in dict
    if "half_precision" in kwargs and kwargs["half_precision"]:
        img = np.minimum(img, np.finfo(np.float16).max)
        result_dict["data"] = img.astype(np.float16)

        #mask = np.minimum(mask, np.finfo(np.float16).max)
        result_dict["seg"] = mask.astype(np.float32)
    else:
        result_dict["data"] = img.astype(np.float32)
        result_dict["seg"] = mask.astype(np.float32)

    if "detection_only" in kwargs and kwargs["detection_only"] is not None:
        detection_only = kwargs["detection_only"]
    else:
        detection_only = True

    # if we consider a simplified task where we want just to detect suspicious
    # lesions regardless of their malignity, we need to set the labels to 1;
    # else load and save the according label
    if detection_only:
        result_dict["label"] = np.float32(1)
    elif "label_type" in kwargs and kwargs["label_type"] is not None:
        if kwargs["label_type"] == "pathology":
            result_dict["label"] = get_patho_label(csv_file, img_key)
        elif kwargs["label_type"] == "birads":
            result_dict["label"] = get_birads_label(csv_file, img_key)
        else:
            raise TypeError("Unknown label type!")
    else:
        raise KeyError("Either detection only flag or label type required!")

    return result_dict

def get_patho_label(csv_file, img_key):
    labels = []

    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the GLOBAL label (here the only one)
        label = data_frame.loc[img_key]['pathology']

        # the label is still a string, but it must be numpy number
        # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
        if label == "MALIGNANT":
            label = np.float32(2)
        else:
            label = np.float32(1)

        # add the label to the dictionary
        labels.append(label)

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the GLOBAL label (here the only one)
            label = data_frame.loc[img_key]['pathology'].iloc[i]

            # the label is still a string, but it must be numpy number
            # define 0 as BENIGN or BENIGN_WITHOUT_CALLBACK and 1 as MALIGNANT
            if label == "MALIGNANT":
                label = np.float32(2)
            else:
                label = np.float32(1)

            # add the label to the dictionary
            labels.append(label)

    if len(labels) > 1:
        # TODO: decide whether this is a suitable approach or not
        label = np.float32(any(labels))
    else:
        label = labels[0]

    return np.float32(label)

def get_birads_label(csv_file, img_key):
    labels = []

    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the GLOBAL label (here the only one)
        label = data_frame.loc[img_key]['assessment']

        # add the label to the dictionary
        labels.append(label)

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the GLOBAL label (here the only one)
            label = data_frame.loc[img_key]['assessment'].iloc[i]

            # add the label to the dictionary
            labels.append(label)

    if len(labels) > 1:
        # TODO: decide whether this is a suitable approach or not
        label = np.max(np.asarray(labels))
    else:
        label = labels[0]

    return  np.clip(label, 1, 5).astype(np.float32)

def get_breast_density(csv_file, img_key):
    breast_density_list = []

    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # extract the GLOBAL label (here the only one)
        breast_density = data_frame.loc[img_key]['breast_density']

        # add the label to the dictionary
        breast_density_list.append(breast_density)

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # extract the GLOBAL label (here the only one)
            breast_density = data_frame.loc[img_key]['breast_density'].iloc[i]

            # add the label to the dictionary
            breast_density_list.append(breast_density)

    if len(breast_density_list) > 1:
        # TODO: decide whether this is a suitable approach or not
        breast_density = np.max(np.asarray(breast_density_list))
    else:
        breast_density = breast_density_list[0]

    return np.float32(breast_density)

def get_labeled_mask(csv_file, img_key, data_path):
    # function to extract the mask and/or the ROIs:
    # this is more complicated, since the csv files are messed up, we need to
    # load BOTH files based on their paths and look up in the meta information
    # which one is which; however, parts of the csv files are correct, so we
    # assume that in general the mask is saved under the path
    # "ROI mask file path"; to be sure, we load the meta information and
    # search for the key word "cropped images" that indicates that this file is
    # the cropped image and not the mask; in this case we switch the paths
    # and load the right image
    masks = []
    rois = []

    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file, index_col='image file path')

    # check how many ROIs are present
    abnormality_id = data_frame.loc[img_key]['abnormality id']

    # if only one is present, abnormality_id will be a np.int64
    if isinstance(abnormality_id, np.int64):
        # get mask path
        roi_path = data_path + \
                   data_frame.loc[img_key]['cropped image file path']
        mask_path = data_path + \
                    data_frame.loc[img_key]['ROI mask file path']

        # sometimes there is a distracting \n that must be removed
        if mask_path[-1] == "\n":
            mask_path = mask_path[:-1]

        if roi_path[-1] == "\n":
            roi_path = roi_path[:-1]

        # find out which one is the actual ROI (and which the mask):
        # Read meta information (keys + values)
        reader = sitk.ImageFileReader()
        reader.SetFileName(mask_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        for k in reader.GetMetaDataKeys():
            if reader.GetMetaData(k) == "cropped images":
                mask_path = roi_path
                break

        # load mask image
        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)
        masks.append(mask)

    # otherwise, it will be a pandas.Series that is iteratable
    else:
        for i in range(len(abnormality_id)):
            # same procedure as above
            # get the ROI and the mask paths
            roi_path = data_path + \
                       data_frame.loc[img_key]['cropped image file path'].iloc[
                           i]
            mask_path = data_path + \
                        data_frame.loc[img_key]['ROI mask file path'].iloc[i]

            # sometimes there is a distracting \n that must be removed
            if mask_path[-1] == "\n":
                mask_path = mask_path[:-1]

            if roi_path[-1] == "\n":
                roi_path = roi_path[:-1]

                # find out which one is the actual ROI (and which the mask):
            # Read meta information (keys + values)
            reader = sitk.ImageFileReader()
            reader.SetFileName(mask_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            for k in reader.GetMetaDataKeys():
                if reader.GetMetaData(k) == "cropped images":
                    mask_path = roi_path
                    break

            # load mask image
            mask_sitk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_sitk)
            masks.append(mask)


    if len(masks) > 1:
        super_mask = masks[0]
        for i in range(1, len(masks)):
            super_mask += masks[i]
    else:
        super_mask = masks[0]

    # clip to be binary mask (not 0 and 255 or so)
    super_mask = np.clip(super_mask, 0, 1)

    return super_mask

def refine_mask(mask, type="calc"):
    if type == "calc":
        #mask = dilation(mask, square(9))
        return mask
    elif type == "mass":
        mask = closing(mask, square(9))
    else:
        raise TypeError("Unknown lesion type!")

    return mask

def load_crops(img_path, data_path, **kwargs):
    # load the sample
    sample = load_sample(img_path, data_path, **kwargs)

    if "crop_size" in kwargs:
        crop_size = kwargs["crop_size"]
    else:
        crop_size = [600, 600]

    # create crop list
    crop_list, _ = utils.create_crops(sample, crop_size)
    image_list = [crop_list[i]["data"] for i in range(len(crop_list))]
    seg_list = [crop_list[i]["seg"] for i in range(len(crop_list))]
    label_list = [crop_list[i]["label"] for i in range(len(crop_list))]

    # the crops that contain only background are useless, therefore we filter
    # them out by thresholding the area that is not zero
    image_list_filtered = []
    seg_list_filtered = []
    label_list_filtered = []

    for j in range(len(crop_list)):
        # check whether at least 25% of the image crop are not zero
        if np.sum((image_list[j][0] > 0).reshape(-1, 1)) \
                < crop_size[1] * crop_size[0] * 0.25:
            continue

        # save the crop
        image_list_filtered.append(image_list[j])

        # save mask
        seg_list_filtered.append(seg_list[j])

        # save labels
        label_list_filtered.append(label_list[j])

    # combine the image list, lable list and bbox list
    crop_list = [{"data": image_list_filtered[j],
                  "seg": np.asarray(seg_list_filtered[j]),
                  "label": label_list_filtered[j]}
                 for j in range(len(image_list_filtered))]

    return crop_list

def load_sample_with_crops(img_path, data_path, **kwargs):
    if "img_shape" in kwargs:
        img_shape = kwargs["img_shape"]
        kwargs["img_shape"] = None
    else:
        img_shape = [1300, 650]

    if "norm" in kwargs:
        norm = kwargs["norm"]
        kwargs["norm"] = False
    else:
        norm = False

    if "half_precision" in kwargs:
        half_prec = kwargs["half_precision"]
        kwargs["half_precision"] = False
    else:
        half_prec = False

    if "num_channel" in kwargs:
        num_channel = kwargs["num_channel"]
    else:
        num_channel = 3

    # load the sample
    sample = load_sample(img_path, data_path, **kwargs)

    # create channel info
    if "channel_mode" in kwargs and kwargs["channel_mode"] is not None:
        ch_info = utils.create_channel_info(sample["data"],
                                          img_shape,
                                          num_channel,
                                          mode=kwargs["channel_mode"])
    else:
        ch_info = utils.create_channel_info(sample["data"],
                                          img_shape,
                                          num_channel,
                                          mode="crops")

    # concatenate crops as channel information
    sample["crops"] = np.concatenate(ch_info, axis=0)

    # resize image
    sample["data"] = utils.resize_image(sample["data"], img_shape)
    sample["seg"] = utils.resize_image(sample["seg"], img_shape)

    # normalize to zero mean and unit variance (if desired)
    if norm:
        img_mean = np.mean(sample["data"][0, :, :])
        img_std = np.std(sample["data"][0, :, :])

        for i in range(sample["data"].shape[0]):
            sample["data"][i, :, :] = (sample["data"][i, :, :] - img_mean) / img_std

        for i in range(sample["crops"].shape[0]):
            sample["crops"][i, :, :] = (sample["crops"][i, :, :] - img_mean) / img_std

    # save all values in dict
    if half_prec:
        sample["data"] = np.float16(sample["data"])
        sample["crops"] = np.float16(sample["crops"])

    return sample

def load_pos_crops(img_path, data_path, **kwargs):
    crop_list = load_crops(img_path, data_path, **kwargs)
    pos_list = []
    for i in range(len(crop_list)):
        if crop_list[i]['label'] > -1:
            pos_list.append(crop_list[i])

    return pos_list

def load_single_pos_crops(img_path, data_path, **kwargs):
    pos_crops = load_pos_crops(img_path, data_path, **kwargs)

    ind = np.random.randint(0, len(pos_crops))

    return pos_crops[ind]

#TODO: either correct or remove, does not work properly (norm issue)
def load_saved_crops(crop_path, data_path, **kwargs):
    # built a dict for all information
    result_dict = {}

    # save the img_path as key for addressing in csv file
    img_key = crop_path.split(data_path)[1].split("/")[1]
    crop_number = \
        crop_path.split(data_path)[1].split("/")[2].split("_")[1].split(".")[0]

    # load image from path and load it as numpy array
    img_sitk = sitk.ReadImage(crop_path)
    img = sitk.GetArrayFromImage(img_sitk)

    # load mask
    mask_sitk = sitk.ReadImage(data_path + "/" + img_key + "/mask_{0}.dcm".format(crop_number))
    mask = sitk.GetArrayFromImage(mask_sitk)

    # normalize to zero mean and unit variance (if desired)
    if "norm" in kwargs and kwargs["norm"]:
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            img[i, :, :] = (img[i, :, :] - np.mean(img[i, :, :])) / np.std(
                img[i, :, :])

    # save all values in dict
    if "half_precision" in kwargs and kwargs["half_precision"]:
        img = np.minimum(img, np.finfo(np.float16).max)
        result_dict["data"] = img.astype(np.float16)
        result_dict["seg"] = mask.astype(np.float32)
    else:
        result_dict["data"] = img.astype(np.float32)
        result_dict["seg"] = mask.astype(np.float32)

    if "detection_only" in kwargs and kwargs["detection_only"] is not None:
        detection_only = kwargs["detection_only"]
    else:
        detection_only = True

    if "label_type" in kwargs and kwargs["label_type"] is not None:
        label_type = kwargs["label_type"]
    else:
        label_type = None

    # read csv file using pandas
    data_frame = pandas.read_csv(kwargs["csv_file"], index_col='img_key')

    # if we consider a simplified task where we want just to detect suspicious
    # lesions regardless of their malignity, we need to set the labels to 1;
    # else load and save the according label
    if detection_only:
        result_dict["label"] = np.float32(1)
    elif label_type is not None:
        if kwargs["label_type"] == "pathology":
            result_dict["label"] = data_frame.loc[img_key]['pathology']
        elif kwargs["label_type"] == "birads":
            result_dict["label"] = data_frame.loc[img_key]['birads']
        else:
            raise TypeError("Unknown label type!")

    return result_dict

def get_crop_paths(crop_dir):
    dir_list = os.listdir(crop_dir)
    crop_list = []
    for i in range(len(dir_list)):
        file_list = os.listdir(crop_dir + "/" + dir_list[i])
        for j in range(len(file_list)):
            if "crop" in file_list[j]:
                crop_list.append(crop_dir + "/" +
                                 dir_list[i] + "/" +
                                 file_list[j])

    return crop_list

def get_paths(data_path, csv_file):
    # define data list
    mammogram_paths = []

    # read csv file using pandas
    data_frame = pandas.read_csv(csv_file)

    # define mammogram list to keep track of what was already considered
    mammogram_list = []

    #TODO: temporary solution, need to be discussed and resolved
    # list of patients to ignore
    patients_train_ignore = ["00059", "00108", "00279", "00304", "00384",
                             "00423", "00436", "00453", "00687", "00694",
                             "00703", "00715", "00765", "00826", "00859",
                             "00915", "00927", "00949", "01048", "01115",
                             "01182", "01243", "01363", "01423", "01486",
                             "01686", "01714", "01757", "01831", "01908",
                             "01946", "01981", "01983", "02033", "02079",
                             "02092"]

    patients_test_ignore = ["00145", "00379", "00381", "00699", "00922",
                            "01378", "01551", "01595"]

    patients_calc_ignore = ["00353"]

    patients_ignore = patients_train_ignore + patients_test_ignore + \
                      patients_calc_ignore

    for i in tqdm(range(data_frame.shape[0])):
        # create unambiguous mammogram key
        patient_id = data_frame.iloc[i]['patient_id'].split("_")[1]
        if patient_id in patients_ignore:
            continue

        patient_view = data_frame.iloc[i]['image view']
        patient_laterality = data_frame.iloc[i]['left or right breast']
        key = patient_id + "_" + patient_view + "_" + patient_laterality

        # check if mammogram already considered, continue if so
        if key in mammogram_list:
            continue

        # otherwise add key to list and load data
        mammogram_list.append(key)
        file_path = data_frame.iloc[i]['image file path']
        mammogram_paths.append(data_path + file_path)

    return mammogram_paths

# wrapper for splitting into train, val and test set
def load_single_set(data_path, csv_file, train_size=0.7, val_size=None,
                    random_state=None):

    paths = get_paths(data_path, csv_file)

    return utils.split_paths_patientwise(paths,
                                         dataset_type = "DDSM",
                                         train_size = train_size,
                                         val_size = val_size,
                                         random_state = random_state)

# wrapper for merging the two distinct datasets into one
def merge_sets(csv_file_train, csv_file_test, new_file_path=None):
    train_frame = pandas.read_csv(csv_file_train)
    test_frame = pandas.read_csv(csv_file_test)

    new_frame = pandas.concat([train_frame, test_frame])

    if new_file_path is None:
        new_frame.to_csv("/home/temp/moriz/data/new_file.csv")
    else:
        new_frame.to_csv(new_file_path)

def generate_crop_directory(img_paths, data_dir, csv_file, save_dir,
                            crop_size=[900, 900], save_prefix=None, **kwargs):
    img_key_list = []
    birads_list = []
    pathology_list = []
    breast_density_list = []

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(len(img_paths))):
        img_key = img_paths[i].split(data_dir)[1]
        file_id = img_key.split("/")[0]
        img_key_list.append(file_id)

        pos_crops = load_pos_crops(img_paths[i],
                                   data_dir,
                                   csv_file=csv_file,
                                   crop_size=crop_size,
                                   **kwargs)

        birads_label = get_birads_label(csv_file, img_key)
        birads_list.append(birads_label)

        patho_label = get_patho_label(csv_file, img_key)
        pathology_list.append(patho_label)

        breast_density = get_breast_density(csv_file, img_key)
        breast_density_list.append(breast_density)

        for j in range(len(pos_crops)):
            sitk_crop = sitk.GetImageFromArray(
                pos_crops[j]["data"].astype(np.int32))
            sitk_mask = sitk.GetImageFromArray(
                pos_crops[j]["seg"].astype(np.int32))

            crop_dir = save_dir + file_id
            if not os.path.isdir(crop_dir):
                os.makedirs(crop_dir)

            sitk.WriteImage(sitk_crop, crop_dir + "/crop_{0}.dcm".format(j))
            sitk.WriteImage(sitk_mask, crop_dir + "/mask_{0}.dcm".format(j))

    data_frame = pandas.DataFrame({'img_key': img_key_list,
                                   'birads': birads_list,
                                   'pathology': pathology_list,
                                   'breast_density': breast_density_list})
    if save_prefix is None:
        csv_save_path = save_dir + "labels.csv"
    else:
        csv_save_path = save_dir + save_prefix + "_labels.csv"

    data_frame.to_csv(csv_save_path)


#old
#---------------------------------------------------------------------------

def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.moveaxis(data, [0, 1, 2], [2, 0, 1])
    return data

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