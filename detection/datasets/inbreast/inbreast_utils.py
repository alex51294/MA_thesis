import SimpleITK as sitk
import numpy as np
import pandas
from skimage import data
from detection.datasets import utils
import os
from skimage.measure import label
from skimage.morphology import closing, square, opening, dilation

import xml.etree.ElementTree as ET
from scipy.interpolate import griddata

def load_sample(img_path, **kwargs):
    # built a dict for all information
    result_dict = {}

    # load image from path and load it as numpy array
    img_sitk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_sitk)

    if "mask_dir" in kwargs:
        png_masks_dir = kwargs["mask_dir"]["png"]
        xml_masks_dir = kwargs["mask_dir"]["xml"]
    else:
        png_masks_dir = "/images/Mammography/INbreast/extras/"
        xml_masks_dir = "/images/Mammography/INbreast/AllXML/"

    if "type" in kwargs and kwargs["type"] in ["mass", "calc", "all"]:
        type = kwargs["type"]
    else:
        raise TypeError("Unsupported lesion type!")

    if "xls_file" in kwargs:
        xls_file = kwargs["xls_file"]
    else:
        xls_file = "/images/Mammography/INbreast/INbreast.xls"

    # extract the image key
    img_key = os.path.split(img_path)[-1].split("_")[0]

    # load mask image
    mask_xml = load_xml_mask(xml_masks_dir, type, img_key,
                             img_shape=(img.shape[1], img.shape[2]))
    mask_binary = load_png_mask(png_masks_dir, type, img_key, img.shape)
    mask = np.clip(mask_xml + mask_binary, 0, 1)

    # correct mask (remove single outliers)
    mask[0, :, :] = closing(mask[0, :, :], square(9))

    # if desired, crop the background and segment the breast
    if "segment" in kwargs and kwargs["segment"]:
        sample = utils.segment_breast({"data": img, "seg": mask})
        # sample = utils.segment_breast_via_hist({"data": img, "seg": mask},
        #                                        thr_x=0.01, thr_y=0.01)
        img = sample["data"]
        mask = sample["seg"]

    # resize image (if desired)
    img_shape = None
    if "img_shape" in kwargs and kwargs["img_shape"] is not None:
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
            ratio = float(
                img.shape[pos_long_side] / img.shape[pos_short_side])
            img_shape = (int(img_shape[0] * ratio), img_shape[0])

        else:
            img_shape = tuple(img_shape)

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

    if img_shape is not None or shape_limit is not None:
        # resize image and mask
        img = utils.resize_image(img, img_shape)
        mask = utils.resize_image(mask, img_shape)

    # normalize to zero mean and unit variance (if desired)
    if "norm" in kwargs and kwargs["norm"]:
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            img[i,:,:] = (img[i,:,:] - np.mean(img[i,:,:])) / np.std(img[i, :, :])

    # adjust the type and save into dictionary
    result_dict['data'] = img.astype(np.float32)

    # add the label to the dictionary
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
        if kwargs["label_type"] == "binary_birads":
            result_dict["label"] = load_binary_birads_label(img_key, xls_file)
        elif kwargs["label_type"] == "birads":
            result_dict["label"] = load_birads_label(img_key, xls_file)
        else:
            raise TypeError("Unknown label type!")

    # add sementation mask
    result_dict['seg'] = np.asarray(mask).astype(np.int)

    return result_dict

def load_birads_label(img_key, xls_file):
    # read csv file using pandas
    data_frame = pandas.read_excel(xls_file, index_col="File Name")

    label = data_frame.loc[np.int32(img_key)]["Bi-Rads"]

    if not isinstance(label, int):
        label = int(label[0])

    label = np.float32(label)

    return label

def load_binary_birads_label(img_key, xls_file):
    birads_label = load_birads_label(img_key, xls_file)

    # define BIRADS label lower or equal 3 as benign, higher as malignant
    if birads_label > 3.0:
        return np.float32(2)

    return np.float32(1)

def load_png_mask(mask_dir, type, img_key, img_shape):
    if type == "mass":
        mask_dir += "MassSegmentationMasks/"
    elif type == "calc":
        mask_dir += "CalcificationSegmentationMasks/"

    mask_path = mask_dir + str(img_key) + "_mask.png"
    if os.path.isfile(mask_path):
        mask = data.load(mask_path, as_gray=True)
        mask = np.expand_dims(mask, 0)
    else:
        mask = np.zeros(img_shape)

    return mask

def load_xml_mask(xml_dir, type, img_key, img_shape):
    # xml file path
    xml_file_path = xml_dir + img_key + ".xml"

    if os.path.isfile(xml_file_path):
        points = parse_mask_file(xml_file_path)
        # print("Number Calcifications: {0}".format(len(points["calc"])))
        # print("Number Masses: {0}".format(len(points["mass"])))
        # print("Number other Lesions: {0}".format(len(points["other"])))

        mask, bboxes = create_mask_from_contour_points(points, img_shape, type)
    else:
        raise FileNotFoundError("XML file does not exist!")

    mask = np.expand_dims(mask, 0)

    return mask

def parse_mask_file(mask_xml_file):
    tree = ET.parse(mask_xml_file)
    dict_list = []
    info_list = []

    for node in tree.iter("dict"):
        dict_list.append(node)

    for i in range(2, len(dict_list)):
        tmp_list = []
        for text in dict_list[i].itertext():
            if not text.startswith("\n"):
                tmp_list.append(text)
        info_list.append(tmp_list)

    calc_points = []
    mass_points = []
    other_points = []

    for h in range(len(info_list)):
        entry = info_list[h]
        number_points = 0
        calc_flag = False
        mass_flag = False
        for i in range(len(entry)):
            if entry[i] == "Calcification" or entry[i] == "Cluster":
                calc_flag = True
            elif entry[i] == "Mass":
                mass_flag = True

            if entry[i] == "NumberOfPoints":
                number_points = np.int32(entry[i + 1])

            if entry[i] == "Point_px":
                points = []
                if calc_flag:
                    for j in range(number_points):
                        text = \
                        entry[i + 1 + j].split("(")[1].split(")")[
                            0].split(", ")
                        points.append(np.float32(text))
                    calc_points.append(points)
                elif mass_flag:
                    for j in range(number_points):
                        text = \
                        entry[i + 1 + j].split("(")[1].split(")")[
                            0].split(", ")
                        points.append(np.float32(text))
                    mass_points.append(points)
                else:
                    for j in range(number_points):
                        text = \
                        entry[i + 1 + j].split("(")[1].split(")")[
                            0].split(
                            ", ")
                        points.append(np.float32(text))
                    other_points.append(points)

    points = {"mass": mass_points,
              "calc": calc_points,
              "other": other_points}

    return points

def parse_muscle_file(muscle_xml_file):
    muscle_points = []

    tree = ET.parse(muscle_xml_file)
    dict_list = []
    info_list = []

    for node in tree.iter("dict"):
        dict_list.append(node)

    for text in dict_list[-1].itertext():
        if not text.startswith("\n"):
            info_list.append(text)

    number_points = 0
    for i in range(len(info_list)):

        if info_list[i] == "NumberOfPoints":
            number_points = np.int32(info_list[i + 1])

        if info_list[i] == "Point_px":
            for j in range(number_points):
                text = \
                    info_list[i + 1 + j].split("(")[1].split(")")[
                        0].split(", ")
                muscle_points.append(np.float32(text))

    return muscle_points


def create_mask_from_contour_points(points, mask_shape, type="mass"):
    mass_points = points["mass"]
    calc_points = points["calc"]
    other_points = points["other"]

    # c, h, w = test_sample["data"].shape
    h, w = mask_shape

    grid_y, grid_x = np.mgrid[0:h, 0:w]
    bboxes = []
    mask = np.zeros((h, w))

    if type == "mass" or type == "all":
        mass_mask = np.zeros((h, w))
        for i in range(len(mass_points)):
            lesion = np.asarray(mass_points[i])
            pos_tl = np.min(lesion, axis=0)
            pos_br = np.max(lesion, axis=0)
            width, height = pos_br - pos_tl
            center = pos_tl + np.asarray([width, height]) / 2
            bboxes.append(
                np.asarray([center[0], center[1], width, height]))
            ind = np.int32(lesion)

            mass_mask = griddata(ind, np.ones_like(ind[:, 0]),
                                 (grid_x, grid_y),
                                 fill_value=0)
        mask = mass_mask

    if type == "calc" or type == "all":
        calc_mask = np.zeros((h, w))
        for i in range(len(calc_points)):
            calc = np.asarray(calc_points[i]).astype(np.int32)
            if len(calc) == 1:
                calc_mask[calc[0][1], calc[0][0]] = 1
                continue
            elif len(calc) < 4:
                for j in range(len(calc)):
                    calc_mask[calc[j][1], calc[j][0]] = 1
                continue
            else:
                calc_mask = griddata(calc, np.ones_like(calc[:, 0]),
                                     (grid_x, grid_y),
                                     fill_value=0)
        mask = mask + calc_mask

    if type == "other" or type == "all":
        for i in range(len(other_points)):
            lesion = np.asarray(other_points[i])
            pos_tl = np.min(lesion, axis=0)
            pos_br = np.max(lesion, axis=0)
            width, height = pos_br - pos_tl
            center = pos_tl + np.asarray([width, height]) / 2
            bboxes.append(
                np.asarray([center[0], center[1], width, height]))
            ind = np.int32(lesion)

            mask = mask + \
                   griddata(ind, np.ones_like(ind[:, 0]), (grid_x, grid_y),
                            fill_value=0)

    if "muscle" in points.keys():
        muscle_points = points["muscle"]

        muscle = np.asarray(muscle_points)
        pos_tl = np.min(muscle, axis=0)
        pos_br = np.max(muscle, axis=0)
        width, height = pos_br - pos_tl
        center = pos_tl + np.asarray([width, height]) / 2
        bboxes.append(
            np.asarray([center[0], center[1], width, height]))
        ind = np.int32(muscle)

        mask = mask + \
               griddata(ind, np.ones_like(ind[:, 0]) * 2,
                        (grid_x, grid_y),
                        fill_value=0)

    return mask, bboxes

def refine_mask(mask, type="calc"):
    if type == "calc":
        #mask = dilation(mask, square(3))
        return mask
    elif type == "mass":
        mask = closing(mask, square(9))
    else:
        raise TypeError("Unknown lesion type!")

    return mask

def load_crops(img_path, **kwargs):
    #load the sample
    sample = load_sample(img_path, **kwargs)

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
        if np.sum((image_list[j][0] > 0).reshape(-1,1)) \
                < crop_size[1] * crop_size[0] * 0.25:
            continue

        # save the crop
        image_list_filtered.append(image_list[j].astype(np.float32))

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

def load_sample_and_crops(img_path, **kwargs):
    if kwargs["crop_size"] != kwargs["img_size"]:
        raise ValueError("Image- and crop size must be identical!")

    img = load_sample(img_path, **kwargs)

    kwargs["img_size"] = None
    crop_list = load_crops(img_path, **kwargs)

    full_list = [img] + crop_list

    return full_list

def load_crop(img_path, **kwargs):
    if "sample_pos" in kwargs:
        sample_pos = kwargs["sample_pos"]
    else:
        sample_pos = 0

    crop_list = load_crops(img_path, **kwargs)

    sample = crop_list[sample_pos]

    return sample

def load_pos_crops(img_path, **kwargs):
    crop_list = load_crops(img_path, **kwargs)
    pos_list = []
    for i in range(len(crop_list)):
        if crop_list[i]['label'] > -1:
            pos_list.append(crop_list[i])

    return pos_list

def load_sample_and_pos_crops(img_path, **kwargs):
    if kwargs["crop_size"] != kwargs["img_size"]:
        raise ValueError("Image- and crop size must be identical!")

    sample = load_sample(img_path, **kwargs)

    kwargs["img_size"] = None
    pos_crop_list = load_pos_crops(img_path, **kwargs)

    full_list = [sample] + pos_crop_list

    return full_list

def get_paths(data_path, xls_file, type="mass"):
    # define data list
    path_list = []

    # change the type-name to fit to the xls sheet
    if type not in ["mass", "calc", "all"]:
        raise ValueError("Unknown type!")

    # read csv file using pandas
    data_frame = pandas.read_excel(xls_file)

    # define mammogram list to keep track of what was already considered
    file_list = os.listdir(data_path)

    # the xls file has two rows more than the actual data
    length_file = data_frame.shape[0] - 2

    for i in range(length_file):
        file_name = data_frame.iloc[i]['File Name']
        if np.isnan(file_name):
            break

        if type == "mass" and data_frame.iloc[i]["Mass"] != "X":
            continue

        if type == "calc" and data_frame.iloc[i]["Micros"] != "X":
            continue

        if type == "all" and (data_frame.iloc[i]["Mass"] != "X" and
                              data_frame.iloc[i]["Micros"] != "X" and
                              data_frame.iloc[i]["Distortion"] != "X" and
                              data_frame.iloc[i]["Asymmetry"] != "X"):
            continue

        file_name = "%d" % file_name

        # intermediate solution (missing masks):
        if file_name == "22614097" or file_name == "22614150":
            continue

        file_path = [i for i in file_list if i.startswith(file_name)]

        path_list.append(data_path + file_path[0])

    return path_list

# wrapper function to split into train, val and test set
def load_single_set(data_path, xls_file, type="mass",
                    train_size=0.7, val_size=None, random_state=None):

    paths = get_paths(data_path, xls_file, type=type)

    return utils.split_paths_patientwise(paths,
                                         dataset_type = "INbreast",
                                         train_size = train_size,
                                         val_size = val_size,
                                         random_state = random_state)

# intermediate function
def get_paths_to_ids(data_path, xls_file, id_list, type="mass"):
    # define data list
    path_list = []

    # change the type-name to fit to the xls sheet
    if type == "mass":
        type = "Mass"
    elif type == "calc":
        type = "Micros"
    else:
        raise ValueError("Unknown type!")

    # read csv file using pandas
    data_frame = pandas.read_excel(xls_file)

    # define mammogram list to keep track of what was already considered
    file_list = os.listdir(data_path)

    # the xls file has two rows more than the actual data
    length_file = data_frame.shape[0] - 2

    for i in range(length_file):
        file_name = data_frame.iloc[i]['File Name']
        if np.isnan(file_name):
            break
        if data_frame.iloc[i][type] != "X":
            continue

        file_name = "%d" % file_name

        # intermediate solution (missing masks):
        if int(file_name) not in id_list:
            continue

        file_path = [i for i in file_list if i.startswith(file_name)]

        path_list.append(data_path + file_path[0])

    return path_list
