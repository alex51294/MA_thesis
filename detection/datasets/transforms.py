# framework for data augmentation; see link in delira for more information
from batchgenerators.transforms import AbstractTransform
from scipy.ndimage.measurements import label as lb
import numpy as np

# not longer required
class MyMirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5
    Args:
        axes (tuple of int): axes along which to mirror
    """
    def __init__(self, axes=(2, 3), data_key="data", bbox_key="bbox"):
        self.data_key = data_key
        self.bbox_key = bbox_key
        self.axes = axes

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        bbox = data_dict.get(self.bbox_key)

        ret_val = self._augment_mirroring(data=data, bbox=bbox, axes=self.axes)

        data_dict[self.data_key] = ret_val[0]
        if bbox is not None:
            data_dict[self.bbox_key] = ret_val[1]

        return data_dict

    def _augment_mirroring(self, data, bbox=None, axes=(2, 3)):
        data = np.copy(data)

        if bbox is not None:
            bbox = np.copy(bbox)

        if (len(data.shape) != 4) and (len(data.shape) != 5):
            raise Exception("Invalid dimension for data and seg. data and seg "
                            "should be [BATCH_SIZE, channels, y, x]")
        batch_size = data.shape[0] # (y,x)
        img_size = data.shape[2:]
        idx = np.arange(batch_size)
        for id in idx:
            # mirror along y-axis
            if 2 in axes and np.random.uniform() < 0.5:
                data[id, :, :, :] = data[id, :, :, ::-1]
                if bbox is not None:
                    bbox[id, :, 0] = np.abs(img_size[1] - bbox[id, :, 0])

            # mirror along x-axis
            if 3 in axes and np.random.uniform() < 0.5:
                data[id, :, :] = data[id, :, ::-1]
                if bbox is not None:
                    bbox[id, :, 1] = np.abs(img_size[0] - bbox[id, :, 1])

        return data, bbox

# Nearly identical to the batchgenerators implementation; however, they force
# the class label to be named "class_targed"; I changed this to label
# (required e.g. for the samplers)
class ConvertSegToBB(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates.

    """

    def __init__(self, dim, margin=None, get_rois_from_seg_flag=False,
                 class_specific_seg_flag=False):
        self.dim = dim
        self.margin = margin
        self.get_rois_from_seg_flag = get_rois_from_seg_flag
        self.class_specific_seg_flag = class_specific_seg_flag

    def __call__(self, **data_dict):
        data_dict = \
            self._convert_seg_to_bounding_box_coordinates(data_dict,
                                                          self.dim,
                                                          self.margin)
        return data_dict

    def _convert_seg_to_bounding_box_coordinates(self,
                                                 data_dict,
                                                 dim,
                                                 margin=None,
                                                 get_rois_from_seg_flag=True,
                                                 class_specific_seg_flag=False):
        '''
        :param data_dict:
        :param dim:
        :param get_rois_from_seg:
        :return: coords (x_center, y_center, width, height)
        '''

        bb_target = []
        roi_masks = []
        roi_labels = []
        out_seg = np.copy(data_dict['seg'])

        for b in range(data_dict['seg'].shape[0]):

            p_coords_list = []
            p_roi_masks_list = []
            p_roi_labels_list = []

            if np.sum(data_dict['seg'][b] != 0) > 0:
                if get_rois_from_seg_flag:
                    clusters, n_cands = lb(data_dict['seg'][b])
                    #data_dict['label'][b] = [data_dict['label'][b]] * n_cands
                else:
                    n_cands = int(np.max(data_dict['seg'][b]))
                    clusters = data_dict['seg'][b]

                rois = np.array([(clusters == ii) * 1 for ii in range(1,
                                                                      n_cands + 1)])  # separate clusters and concat
                for rix, r in enumerate(rois):
                    if np.sum(
                            r != 0) > 0:  # check if the lesion survived data augmentation
                        seg_ixs = np.argwhere(r != 0)
                        coord_list = [np.min(seg_ixs[:, 1]) - 1,
                                      np.min(seg_ixs[:, 2]) - 1,
                                      np.max(seg_ixs[:, 1]) + 1,
                                      np.max(seg_ixs[:, 2]) + 1]
                        if dim == 3:
                            coord_list.extend([np.min(seg_ixs[:, 3]) - 1,
                                               np.max(seg_ixs[:, 3]) + 1])

                        p_coords_list.append(coord_list)
                        p_roi_masks_list.append(r)
                        # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                        # also patient wide, this assignment is not dependent on patch occurrances.
                        #p_roi_labels_list.append(data_dict['label'][b][rix] + 1)
                        p_roi_labels_list.append(data_dict['label'][b])

                    if class_specific_seg_flag:
                        out_seg[b][data_dict['seg'][b] == rix + 1] = \
                        data_dict['label'][b][rix] + 1

                if not class_specific_seg_flag:
                    out_seg[b][data_dict['seg'][b] > 0] = 1

                bb_target.append(np.array(p_coords_list))
                roi_masks.append(
                    np.array(p_roi_masks_list).astype('uint8'))
                roi_labels.append(np.array(p_roi_labels_list))


            else:
                bb_target.append(np.array([[0, 0, 1, 1]]))
                roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
                roi_labels.append(np.array([0]))


        # if get_rois_from_seg_flag:
        #     data_dict.pop('label', None)

        # add margin if desired
        if margin is not None:
            if margin < 0 or margin > 1:
                ValueError("Margin must be a value in [0,1]!")
            for i in range(len(bb_target)):
                #bb_target = np.asarray(bb_target)
                height = bb_target[i][:, 2] - bb_target[i][:, 0]
                width = bb_target[i][:, 3] - bb_target[i][:, 1]

                offset = np.asarray([np.floor(height * margin),
                                     np.floor(width * margin)])

                # make sure bounds are valid
                bb_target[i][:, 0] = np.clip(
                    np.floor(bb_target[i][:, 0] - height * margin * 0.5),
                    0, data_dict["seg"].shape[2] - 1)
                bb_target[i][:, 1] = np.clip(
                    np.floor(bb_target[i][:, 1] - width * margin * 0.5),
                    0, data_dict["seg"].shape[3] - 1)
                bb_target[i][:, 2] = np.clip(
                    np.floor(bb_target[i][:, 2] + height * margin * 0.5),
                    0, data_dict["seg"].shape[2] - 1)
                bb_target[i][:, 3] = np.clip(
                    np.floor(bb_target[i][:, 3] + width * margin * 0.5),
                    0, data_dict["seg"].shape[3] - 1)

        # if used on cropped masks, small overlapping bbox might appear
        # that should be removed
        # reference = bb_target[0]
        # num_bbox_params = len(bb_target)
        # for i in range(num_bbox_params):
        #     area = (bb_target[i][3] -bb_target[i][1]) * \
        #            (bb_target[i][2] - bb_target[i][0])
        #     if area > (reference[3] - reference[1]) * \
        #             (reference[2] - reference[0]):
        #         reference = bb_target[i]
        #
        # i = 0
        # while i < num_bbox_params:
        #     if (bb_target[i][0] > reference[0] and \
        #         bb_target[i][1] > reference[1]) or \
        #             (bb_target[i][2] < reference[2] and \
        #              bb_target[i][3] < reference[3]):
        #         bb_target.pop(i)
        #         num_bbox_params -= 1
        #         i = 0
        #         continue
        #     i += 1

        # # convert into numpy and save in [x_center, y_center, width ,hight] format
        #bb_target = np.asarray(bb_target)
        #print(bb_target)
        #tmp = np.copy(bb_target).astype(np.float32)

        for i in range(len(bb_target)):
            size = bb_target[i][:, 2:] - bb_target[i][:, 0:2]
            center = np.floor(bb_target[i][:, 0:2] + size / 2)
            bb_target[i][ :, 0] = center[:, 1]
            bb_target[i][:, 1] = center[:, 0]
            bb_target[i][:, 2] = size[:, 1]
            bb_target[i][:, 3] = size[:, 0]
            bb_target[i] = bb_target[i].astype(np.float32)

        data_dict['bboxes'] = np.array(bb_target)
        data_dict['roi_masks'] = np.array(roi_masks)
        data_dict['roi_labels'] = np.array(roi_labels)
        data_dict['seg'] = out_seg

        return data_dict

class Rot90Transform(AbstractTransform):
    def __init__(self, num_rot=(1, 2, 3), axes=(0, 1, 2), data_key="data",
                 label_key="seg", p_per_sample=0.3):
        """
        :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
        :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
        :param data_key:
        :param label_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.label_key = label_key
        self.data_key = data_key
        self.axes = axes
        self.num_rot = num_rot

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                d = data[b]
                if seg is not None:
                    s = seg[b]
                else:
                    s = None
                d, s = self._augment_rot90(d, s, self.num_rot, self.axes)
                data[b] = d
                if s is not None:
                    seg[b] = s

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

    def _augment_rot90(self, sample_data, sample_seg, num_rot=(1, 2, 3),
                      axes=(0, 1, 2)):
        """
        :param sample_data:
        :param sample_seg:
        :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
        :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
        :return:
        """
        num_rot = np.random.choice(num_rot)
        axes = np.random.choice(axes, size=2, replace=False)
        axes.sort()
        axes = [i + 1 for i in axes]
        sample_data = np.rot90(sample_data, num_rot, axes)
        if sample_seg is not None:
            sample_seg = np.rot90(sample_seg, num_rot, axes)
        return sample_data, sample_seg
