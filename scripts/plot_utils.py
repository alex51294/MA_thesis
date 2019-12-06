import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from tqdm import tqdm
import numpy as np
import os

import detection.datasets.utils as utils
from sklearn.metrics import precision_recall_curve

# maybe unnecessary, but did not work otherwise
backend = matplotlib.get_backend()
if backend != 'TkAgg':
    matplotlib.use('module://backend_interagg')

def plot_mammogram(image, mask=None, margin=None, annotation=None,
                   plot_mask=False, rel_figsize=0.25,
                   image_save=False,
                   image_save_dir="/home/temp/moriz/validation/",
                   save_suffix=None, format="pdf"):

    if len(image.shape) == 3:
        c, h, w = image.shape
        figsize = rel_figsize * (w / 100), rel_figsize * (h / 100)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image[0, :, :], cmap='Greys_r')
    else:
        h, w = image.shape
        figsize = rel_figsize * (w / 100), rel_figsize * (h / 100)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap='Greys_r')

    if mask is not None:
        if plot_mask:
            if len(mask.shape) == 3:
                ax.imshow(mask[0, :, :], cmap='hot', alpha=0.5)
            elif len(mask.shape) == 2:
                ax.imshow(mask, cmap='hot', alpha=0.5)
            else:
                raise ValueError("Too many input dimensions!")

        bbox = utils.bounding_box(mask, margin)
        number_bbox = len(bbox)
        #print("Number bboxes: {0}".format(number_bbox))

        for k in range(number_bbox):
            pos = tuple(bbox[k][0:2])
            width = bbox[k][2]
            height = bbox[k][3]

            pos = (pos[0] - np.floor(width / 2), pos[1] - np.floor(height / 2))

            # Create a Rectangle patch
            rect = patches.Rectangle(pos, width, height, linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            if annotation is not None:
                if isinstance(annotation, np.ndarray):
                    ax.annotate("{0}".format(annotation[k]), pos,
                                fontsize=10,
                                color="b",
                                xytext=(pos[0] + 10, pos[1] - 10))
                else:
                    ax.annotate("{0}".format(annotation), pos,
                                fontsize=10,
                                color="b",
                                xytext=(pos[0] + 10, pos[1] - 10))

    if image_save:
        if save_suffix is None:
            save_name = "/mammogram." + format
        else:
            save_name = "/mammogram_" + save_suffix + "." + format
        plt.savefig(image_save_dir + save_name, dpi='figure',
                    format=format)
    plt.show()
    plt.close(fig)

def plot_dataset(dataset, plot_crops=False, plot_pos_crops=False,
                 crop_size = [600, 600], **kwargs):

    for i in tqdm(range(len(dataset))):
        # plot image
        plot_mammogram(dataset[i]["data"], dataset[i]["seg"],
                       annotation= dataset[i]["label"], **kwargs)

        # create and plot crops if desired
        if plot_crops or plot_pos_crops:
            crop_list, corner_list = utils.create_crops(dataset[i],
                                                        crop_size=crop_size)

            # utils.show_crop_distribution(sample_data["data"], crop_size,
            #                              corner_list, heatmap=True)

            # iterate over crops
            for j in tqdm(range(0, len(crop_list))):
                crop_image = crop_list[j]['data']
                crop_mask = crop_list[j]['seg']
                crop_label = crop_list[j]["label"]

                if plot_pos_crops:
                    if crop_label > 0:
                        plot_mammogram(crop_image, crop_mask,
                                       annotation=crop_label, **kwargs)
                else:
                    plot_mammogram(crop_image, crop_mask,
                                   annotation=crop_label, **kwargs)


def plot_ap(ap_list, models, image_save=False,
            image_save_dir="/home/temp/moriz/validation/",
            save_suffix=None, format="pdf"):
    start_epoch, end_epoch, step_size = models

    tmp = np.asarray(ap_list)
    best_epochs = []
    for i in range(5):
        be = np.argmax(tmp)
        best_epochs.append(be)
        tmp[be] = 0

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel("Model")
    plt.xlim(start_epoch - step_size, end_epoch + step_size)
    plt.ylabel("AP")
    plt.ylim(0, 1.1)
    plt.title("AP values for the examined models")
    ax.plot(
        np.arange(start_epoch, end_epoch + step_size, step_size),
        np.asarray(ap_list), "rx-")
    for i in range(5):
        circle = patches.Ellipse((best_epochs[i] * step_size + start_epoch,
                                 ap_list[best_epochs[i]]),
                                 width=step_size, height=0.05,
                                 edgecolor="b", linewidth=1, facecolor="none")
        ax.add_patch(circle)
    plt.gca().legend(["AP values"] + \
                     ["Best epoch {0}".format(best_epochs[0] * step_size + start_epoch)] + \
                     ["{0}. best epoch: {1}".format(i+1, best_epochs[i]*step_size + start_epoch)
                     for i in range(1, 5)])

    if image_save:
        if save_suffix is None:
            save_name = "/AP." + format
        else:
            save_name = "/AP_" + save_suffix + "." + format
        plt.savefig(image_save_dir + save_name, dpi='figure',
                    format=format)
    plt.show()
    plt.close(fig)


def plot_precion_recall_curve(precision_steps,
                              recall_steps = np.linspace(.0, 1, 101, endpoint=True),
                              ap_value = None,
                              image_save=False,
                              image_save_dir="/home/temp/moriz/validation/",
                              save_suffix=None, format="pdf"):

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel("Recall")
    plt.xlim(0, 1)
    plt.ylabel("Precision")
    plt.ylim(0, 1.1)
    plt.title("Precision-recall curve")
    ax.plot(recall_steps, precision_steps, "rx-")

    if ap_value is not None:
        ax.legend(["AP: {:.4f}".format(ap_value)])

    if image_save:
        if save_suffix is None:
            save_name = "/PR_curve." + format
        else:
            save_name = "/PR_curve_" + save_suffix + "." + format
        plt.savefig(image_save_dir + save_name, dpi='figure',
                    format=format)

    plt.show()
    plt.close(fig)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          image_save=False,
                          image_save_dir="/home/temp/moriz/validation/",
                          save_suffix=None,
                          save_format="pdf"
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Confusion matrix'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if image_save:
        if save_suffix is None:
            save_name = "/confusion_matrix." + save_format
        else:
            save_name = "/confusion_matrix_" + save_suffix + "." + save_format
        plt.savefig(image_save_dir + save_name, dpi='figure',
                    format=save_format)

    plt.show()
    plt.close(fig)

def plot_f1(f1_list, confidence_values, models=None,
            image_save=False, image_save_dir="/home/temp/moriz/validation/",
            plot_average=False, plot_maximum=False, plot_all=False,
            format="pdf"):
    # convert list into numpy array
    f1_list = np.transpose(np.asarray(f1_list))

    if models is not None:
        # determine how many models
        start_epoch, end_epoch, step_size = models
        models = np.arange(start_epoch, end_epoch + step_size,
                           step_size)

        if plot_all:
            # define how many frocs shall be plotted in one graph
            num_graphs = 5
            for i in range(int(np.ceil(len(models) / num_graphs))):
                if (i + 1) * num_graphs > len(models):
                    current_f1_list = f1_list[:, i * num_graphs:]
                    current_models = models[i * num_graphs:]
                else:
                    current_f1_list = f1_list[:,
                                      i * num_graphs:num_graphs * (i + 1)]
                    current_models = models[i * num_graphs: (i + 1) * num_graphs]

                fig, ax = plt.subplots(num=i, figsize=(15, 10))
                plt.xlabel("thresholds")
                plt.ylabel("F1")
                plt.ylim(0, 1.1)
                plt.title(
                    "F1 values at the single thresholds for the examined models")
                ax.plot(confidence_values, current_f1_list, "x-")
                plt.gca().legend(
                    ["epoch_" + str(j) for j in current_models])
                if image_save:
                    plt.savefig(
                        image_save_dir + "/F1s_{0}".format(i) + "." + format,
                        dpi='figure', format=format)
                plt.show()
                plt.close(fig)

        if plot_average or plot_all:
            # averaged over the single thresholds for all models
            fig, ax = plt.subplots(figsize=(15, 10))
            plt.xlabel("Model", fontsize=20)
            plt.ylabel("F1", fontsize=20)
            plt.ylim(0, 1.1)
            plt.title("F1 average values for the examined models", fontsize=25)
            plt.tick_params(axis="both", labelsize=15)
            # scale = confidence_values.reshape(-1,1).repeat(f1_list.shape[1], axis=1)
            # ax.plot(np.arange(start_epoch, end_epoch + step_size, step_size),
            #         np.sum(f1_list * scale, axis=0) / np.float32(len(confidence_values)),
            #         "rx-")
            average_values = np.sum(f1_list, axis=0) / np.float32(len(confidence_values))

            tmp = np.copy(average_values)
            best_epochs = []
            for i in range(5):
                be = np.argmax(tmp)
                best_epochs.append(be)
                tmp[be] = 0
            ax.plot(np.arange(start_epoch, end_epoch + step_size, step_size),
                    average_values,
                    "rx-")
            for i in range(5):
                circle = patches.Ellipse((best_epochs[i] * step_size + start_epoch,
                                          average_values[best_epochs[i]]),
                                         width=step_size, height=0.05,
                                         edgecolor="b", linewidth=1,
                                         facecolor="none")
                ax.add_patch(circle)
            plt.gca().legend(["Averaged F1 scores"] + \
                             ["Best epoch {0}".format(
                                 best_epochs[0] * step_size + start_epoch)] + \
                             ["{0}. best epoch: {1}".format(i + 1, best_epochs[
                                 i] * step_size + start_epoch)
                              for i in range(1, 5)])

            if image_save:
                plt.savefig(image_save_dir + "/F1_average." + format,
                            dpi='figure', format=format)
            plt.show()
            plt.close(fig)

        if plot_maximum or plot_all:
            maximum_values = np.max(f1_list, axis=0)
            max_thr = confidence_values[np.argmax(f1_list, axis=0)]

            plt.figure(figsize=(15, 15))
            plt.title("F1 maximum values for the examined models", fontsize=25)
            plt.tick_params(axis="both", labelsize=15)

            plt.subplot(211)
            plt.xlabel("Model", fontsize=20)
            plt.ylabel("max. F1 value", fontsize=20)
            plt.ylim(0, 1.1)
            plt.plot(np.arange(start_epoch, end_epoch + step_size, step_size),
                     maximum_values,
                     "rx-")

            plt.subplot(212)
            plt.xlabel("Model", fontsize=20)
            plt.ylabel("Confidence score at max. F1 value", fontsize=20)
            plt.ylim(0, 1.1)
            plt.plot(np.arange(start_epoch, end_epoch + step_size, step_size),
                     max_thr,
                     "rx-")
            if image_save:
                plt.savefig(image_save_dir + "/F1_max." + format, dpi='figure',
                            format=format)
            plt.show()
            plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.xlabel("thresholds", fontsize=20)
        plt.ylabel("F1", fontsize=20)
        plt.ylim(0, 1.1)
        plt.title(
            "F1 values at the single thresholds for the examined model",
            fontsize=25)
        plt.tick_params(axis="both", labelsize=15)
        ax.plot(confidence_values, f1_list, "x-")
        if image_save:
            plt.savefig(
                image_save_dir + "/F1." + format,
                dpi='figure', format=format)
        plt.show()
        plt.close(fig)
        # f1_average = np.sum(f1_list * confidence_values, axis=0) / np.float32(len(confidence_values))
        # print("F1 average: {0}".format(f1_average))

def plot_frocs(froc_tpr_list, froc_fppi_list, models=None,
               image_save=False, image_save_dir="/home/temp/moriz/validation/",
               image_save_name = None,
               left_range=1e-1, right_range=1e2, grid=True, format="pdf",
               ylim=(0, 1.1),
               legend=None,
               legend_position=2,
               title=None):
    # convert list into numpy array
    froc_fppi = np.transpose(np.asarray(froc_fppi_list))
    froc_tpr = np.transpose(np.asarray(froc_tpr_list))

    if models is not None:
        # determine how many models
        start_epoch, end_epoch, step_size = models
        models = np.arange(start_epoch, end_epoch + step_size,
                           step_size)

        # define how many frocs shall be plotted in one graph
        num_frocs = 5
        for i in range(int(np.ceil(len(models) / num_frocs))):
            if (i + 1) * num_frocs > len(models):
                current_froc_fppi = froc_fppi[:, i * num_frocs:]
                current_froc_tpr = froc_tpr[:, i * num_frocs:]
                current_models = models[i * num_frocs:]
            else:
                current_froc_fppi = froc_fppi[:,
                                    i * num_frocs:num_frocs * (i + 1)]
                current_froc_tpr = froc_tpr[:,
                                   i * num_frocs:num_frocs * (i + 1)]
                current_models = models[
                                 i * num_frocs: (i + 1) * num_frocs]

            fig, ax = plt.subplots(num=i, figsize=(15, 10))
            plt.xlabel("FPPI", fontsize=20)
            plt.ylabel("TPR", fontsize=20)
            plt.ylim(0, 1.1)
            plt.xlim(left=left_range, right=right_range)
            plt.xscale("log")
            plt.title(
                "FROCs for the examined models, plot number {0}".format(
                    i), fontsize=25)
            plt.tick_params(axis="both", labelsize=15)
            ax.plot(current_froc_fppi, current_froc_tpr, "x-")
            plt.gca().legend(
                ["epoch_" + str(j) for j in current_models])
            if image_save:
                plt.savefig(
                    image_save_dir + "/FROCs_{0}".format(i) + "." + format,
                    dpi='figure', format=format)
            plt.show()
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.xlabel("FPPI", fontsize=30)
        plt.ylabel("TPR", fontsize=30)
        plt.ylim(ylim)
        plt.xlim(left=left_range, right=right_range)
        plt.xscale("log")
        plt.tick_params(axis="both", labelsize=20)
        if title is not None:
            if isinstance(title, str):
                plt.title(title, fontsize=25)
            elif isinstance(title, bool):
                plt.title("FROC for the examined models", fontsize=25)

        ax.plot(froc_fppi, froc_tpr, "x-")
        if grid:
            #plt.grid(b=True, which="both")
            plt.grid(b=True)

        if legend is not None:
            plt.gca().legend(legend, fontsize=20, loc=legend_position)
        if image_save:
            if image_save_name is None:
                plt.savefig(image_save_dir + "/FROC." + format,
                            dpi='figure', format=format)
            else:
                plt.savefig(image_save_dir + "/" + image_save_name + "." + format,
                            dpi='figure', format=format)
        plt.show()
        plt.close(fig)

def plot_roc(fpr, tpr, legend=None, image_save=False, image_save_dir=None,
             image_save_name=None, format="pdf"):
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel("FPR", fontsize=20)
    plt.ylabel("TPR", fontsize=20)
    plt.ylim(0, 1.1,)
    plt.xlim(0., 1.1)
    plt.tick_params(axis="both", labelsize=15)
    plt.title("ROC for the examined model", fontsize=25)
    ax.plot(fpr, tpr, "x-")
    if legend is not None:
        plt.gca().legend(legend)
    if image_save:
        if image_save_name is None:
            plt.savefig(image_save_dir + "/ROC." + format,
                        dpi='figure', format=format)
        else:
            plt.savefig(image_save_dir + "/" + image_save_name + "." + format,
                        dpi='figure', format=format)
    plt.show()
    plt.close(fig)

def show_crop_subdivision(image, crop_size, corner_list, heatmap = False,
                          rel_figsize=0.25,
                          image_save=False,
                          image_save_dir="/home/temp/moriz/validation/",
                          image_save_prefix=None,
                          format="pdf"):
    """

    :param image: image, whose subdivision shall be evaluated; either of the
                    form (c, y, x) or (y,x)
    :param crop_size: crop size that was used to divide the image into crops;
                    must be of the form (y,x)
    :param corner_list: corner list, discribing the top left corner of each
                    crop with respect to the image; each position is of the
                    form (x,y) (ATTENTION!)
    :param heatmap: boolean value, states whether a heatmap-format shall be
                    shown
    :return: none
    """

    # plot image to visualize crop subdivision (if desired)
    if len(image.shape) == 3:
        c, h, w = image.shape
        figsize = rel_figsize * (w / 100), rel_figsize * (h / 100)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image[0, :, :], cmap='Greys_r')
    else:
        h, w = image.shape
        figsize = rel_figsize * (w / 100), rel_figsize * (h / 100)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap='Greys_r')

    color = ["g", "b", "y", "c", "m"]

    for i in range(len(corner_list)):
        plt.plot(corner_list[i][0], corner_list[i][1], 'r.')

    for i in range(len(corner_list)):
        rect = patches.Rectangle((corner_list[i][0], corner_list[i][1]),
                                 crop_size[1],
                                 crop_size[0],
                                 linewidth=1,
                                 edgecolor=color[i % 5],
                                 facecolor='none')
        ax.add_patch(rect)

    if not os.path.isdir(image_save_dir):
        os.makedirs(image_save_dir)

    if image_save:
        if image_save_prefix is not None:
            save_name = image_save_dir + "/" + \
                        image_save_prefix + "_image_cropped." + format
        else:
            save_name = image_save_dir + "/image_cropped." + format
        plt.savefig(save_name, dpi='figure', format=format)
    plt.show()
    plt.close(fig)

    if heatmap:
        heatmap = np.zeros((h,w))
        for i in range(len(corner_list)):
            heatmap[corner_list[i][1]:corner_list[i][1] + crop_size[0],
            corner_list[i][0]:corner_list[i][0] + crop_size[1]] += 1


        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(heatmap, cmap='hot')
        plt.colorbar(img)

        if image_save:
            if image_save_prefix is not None:
                save_name = image_save_dir + "/" + \
                            image_save_prefix + "_heatmap." + format
            else:
                save_name = image_save_dir + "/heatmap." + format
            plt.savefig(save_name, dpi='figure', format=format)

    plt.show()
    plt.close(fig)


def plot_heatmap(heatmap):
    fig, ax = plt.subplots(figsize=(15, 10))
    img = ax.imshow(heatmap, cmap="jet")
    plt.colorbar((img))
    plt.show()
    plt.close(fig)


# under development
# def plot_afroc(afroc_tpr_list, afroc_fpr_list, models,
#                image_save_dir, image_save=False):
#     # determine how many models
#     start_epoch, end_epoch, step_size = models
#     models = np.arange(start_epoch, end_epoch + step_size,
#                        step_size)
#
#     # convert list into numpy array
#     froc_tpr = np.transpose(np.asarray(afroc_tpr_list))
#     afroc_fpr = np.transpose(np.asarray(afroc_fpr_list))
#
#     # define how many frocs shall be plotted in one graph
#     num_frocs = 5
#     for i in range(int(np.ceil(len(models) / num_frocs))):
#         if (i + 1) * num_frocs > len(models):
#             current_afroc_fpr = afroc_fpr[:, i * num_frocs:]
#             current_froc_tpr = froc_tpr[:, i * num_frocs:]
#             current_models = models[i * num_frocs:]
#         else:
#             current_afroc_fpr = afroc_fpr[:,
#                                 i * num_frocs:num_frocs * (i + 1)]
#             current_froc_tpr = froc_tpr[:,
#                                i * num_frocs:num_frocs * (i + 1)]
#             current_models = models[
#                              i * num_frocs: (i + 1) * num_frocs]
#
#         fig, ax = plt.subplots(num=i, figsize=(15, 10))
#         plt.xlabel("AFROC FPR")
#         plt.ylabel("TPR")
#         plt.title(
#             "AFROCs for the examined models, plot number {0}".format(
#                 i))
#         ax.plot(current_afroc_fpr, current_froc_tpr, "x-")
#         plt.gca().legend(
#             ["epoch_" + str(j) for j in current_models])
#         if image_save:
#             plt.savefig(
#                 image_save_dir + "/AFROCs_{0}".format(i) + ".svg",
#                 dpi='figure', format='svg')
#         plt.show()
#         plt.close(fig)

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