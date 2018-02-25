import numpy as np
import package_KITTI
import os
from PIL import Image, ImageDraw, ImageColor, ImageFont
from prettytable import PrettyTable

KITTI_CLASSES_LIST = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                     'Cyclist', 'Tram', 'Misc', 'DontCare']

KITTI_CLASSES = package_KITTI.KITTI_CLASSES

def loadFiles():
    true_label_file = "KITTI-dev.npz"
    predicted_label_file = "./images/out/predicted_labels.npz"
    y = np.load(true_label_file)
    yhat = np.load(predicted_label_file)
    CalcMetrics(y, yhat)


def CalcMetrics(y, yhat):
    """
    Calculate Precision and Recall
    :param y: True label data. numpy.lib.npyio.NpzFile containing the following variables:
        'image_size': Size of resized images (to correct rectification)
        'image_path': Path to the directory containing the images
        'image_labels': (mx1) list of dictionaries with true image label data. Contains the following fields:
            'objects': (Cx5) numpy array of C detected objects containing [class (integer), x1, y1, x2, y2]
            'classes': (Cx1) list of strings of the detected classes (KITTI)
            'file': (Cx1) list of strings of the file names, with image extension
    :param yhat: Predicted label data. numpy.lib.npyio.NpzFile containing the following variables:
            'image_labels': (mx1) list of dictionaries with true image label data. Contains the following fields:
            'objects': (Cx5) numpy array of C detected objects containing [class (integer), x1, y1, x2, y2]
            'classes': (Cx1) list of strings of the detected classes (KITTI)
            'file': (Cx1) list of strings of the file names, with image extension
            'scores': (Cx1) list of floats with the confidence scores for the detections
    :return:
    """

    # Parameters
    iou_threshold = 0.5      # Threshold for an accurate localization
    plot = False             # Plot the images with bounding boxes
    car_forgiveness = True   # Allows DontCare true labels to be accepted as cars
    num_images = 100         # Number of images to analyze

    # Important vars
    num_classes = len(package_KITTI.KITTI_CLASSES)

    # Get image size
    imsize_true = y['image_size']

    # Get image paths
    path_predict = str(y['image_path'])
    path_true = y['image_path']

    # Instatiate the plotter object
    plotter = BoxPlotter(path_predict, imsize_true)

    # Extract image labels
    y = y['image_labels']
    yhat = yhat['image_labels']

    # Get number of images in each set
    m_predict = len(yhat)
    m_true = len(y)

    # Get file names for each set
    names_true = np.array([image['file'] for image in y])
    names_predict = np.array([image['file'] for image in yhat])

    # Make the two lists the same length and ordered identically
    if m_true >= m_predict:
        inds = np.isin(names_true, names_predict)
        names_true = names_true[inds]
        if len(names_true) != m_predict:
            raise Exception("Unique predictions exist")
        m_true = len(names_true)
        sort_true = np.argsort(names_true)
        sort_predict = np.argsort(names_predict)
        y = y[sort_true]
        yhat = yhat[sort_predict]
    # TODO: Add option when there are more predictions than true labels

    # Set up variables
    m = np.minimum(len(y), num_images)
    TP = np.zeros(num_classes)  # True positive (corrected)
    FP = np.zeros(num_classes)  # False positive (detected but incorrect)
    FN = np.zeros(num_classes)  # False negative (not detected)
    CN = np.zeros(num_classes)  # Count of true labels

    # Loop over all of the images
    for i in range(m):
        # Extract true label data
        objects = y[i]['objects']
        num_objects = objects.shape[0]
        c_true = objects[:, 0].astype(int)  # class (int, KITTI)
        bb_true = objects[:, 1:]
        matches = np.zeros(num_objects)

        result = np.zeros(yhat[i]['objects'].shape[0])  # track if the box was FP (0) or TP (1)

        # Loop over each predicted detection
        for j, label in enumerate(yhat[i]['objects']):
            chat_coco = label[0:1].astype(int)  # predicted class (int, COCO)
            chat = COCO2KITTI(chat_coco)        # convert to KITT classes TODO: change detection to detect KITTI classes
            bbhat = label[1:]                   # predicted bounding box

            # Match the bounding box with the bounding box with the greatest IOU
            ious = [iou(bbhat, bb)for bb in bb_true]
            match_ind = int(np.argmax(ious))  # gives the index of the true label that best matches the prediction
            matches[match_ind] += 1  # count how many times true label is matched

            # Check if it matches the correct class and greater than threshold
            if c_true[match_ind] == chat and ious[match_ind] > iou_threshold:
                TP[chat] += 1
                result[j] = 1
            elif (car_forgiveness  # allows true "DontCare" labels to accept "Car" as true
                    and c_true[match_ind] == KITTI_CLASSES["DontCare"]
                    and chat == KITTI_CLASSES["Car"]):
                TP[chat] += 1
                result[j] = 1
            else:
                FP[chat] += 1
                result[j] = 0

        # Add result to image for plotting purposes
        yhat[i]['result'] = result

        # Count all true boxes that were never matched as false negatives
        y[i]['result'] = matches > 0
        for j in (matches == 0).nonzero():
            FN[c_true[j]] += 1

        # Get true counts for stats
        for c in c_true:
            CN[c] += 1

        if np.any(matches > 1):
            print("Some boxes were detected more than once")

        # Plot
        if plot:
            plotter.comparison(y[i], yhat[i])

    # Calculate precision and recall
    eps = 1e-9
    mAP = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    header = list(KITTI_CLASSES_LIST)
    total_detections = TP + FP

    header.insert(0, "")
    T = PrettyTable(header)
    T.add_row(np.hstack(("TP", TP)))
    T.add_row(np.hstack(("FP", FP)))
    T.add_row(np.hstack(("FN", FN)))
    T.add_row(np.hstack(("TD", total_detections)))
    T.add_row(np.hstack(("TL", CN)))
    T.add_row(np.hstack(("mAP", np.round(mAP, 2))))
    T.add_row(np.hstack(("RCL", np.round(recall, 2))))
    print(T)


class BoxPlotter:
    """
    Object to plot bounding boxes for predicted and true labels
    """
    def __init__(self, image_folder, image_size):
        """
        :param image_folder: folder containing the images
        :param image_size: size of the resized images
        """
        # Set label font
        self.font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image_size[1] + 0.5).astype('int32'))
        self.image_size = image_size

        # Set bounding box colors
        self.color_pred = [ImageColor.getrgb('red'),  # False Positive
                      ImageColor.getrgb('green')]  # True Positive
        self.color_true = [ImageColor.getrgb('orange'),  # False Negative
                      ImageColor.getrgb('blue')]  # Detected true box
        self.image_folder = image_folder

    def comparison(self, y, yhat):
        """
        Plots both predicted and true bounding boxes for comparison
        :param y: dictionary of a true label data. One entry of the "y" input to the CalcMetrics function
        :param yhat: dictionary of predicted label data
        :return: Nothing. Displays a plot to the screen
        """
        # Open image and set up drawing variables
        image_path = os.path.join(self.image_folder, yhat['file'])
        image = Image.open(image_path)
        image = image.resize(self.image_size, Image.BICUBIC)
        draw = ImageDraw.Draw(image)

        # Plot the boxes
        self.truth_boxes(draw, y)
        self.prediction_boxes(draw, yhat)
        image.show()

        # Cleanup
        del draw

    def prediction_boxes(self, draw, yhat):
        """
        Plot bounding boxes for predicted labels. Box label includes class name and confidence
        :param draw: PIL.ImageDraw.draw object
        :param yhat: dictionary of predicted detection data
        :return: Nothing
        """
        # Loop over each object detected in the image
        num_detections = yhat['objects'].shape[0]
        for j in range(num_detections):
            # Extract out import info from dictionary
            classname = yhat['classes'][j]
            score = yhat['scores'][j]
            result = int(yhat['result'][j])
            box = yhat['objects'][j, 1:]

            # Set label and color
            label = '{} {:.2f}'.format(classname, score)
            color = self.color_pred[result]

            # Plot the boxes
            self.plot_box(draw, box, label, color)

    def truth_boxes(self, draw, y):
        """
        Plot bounding boxes for true labels. Box label includes class name and (Truth)
        :param draw: PIL.ImageDraw.draw object
        :param y: dictionary of true detection label data
        :return: Nothing
        """
        # Loop over each object detected in the image
        num_true = y['objects'].shape[0]
        for j in range(num_true):
            # Extract important info from dictionary
            classname = y['classes'][j]
            result = y['result'][j]
            box = y['objects'][j, 1:]

            # Set label and color
            label = '{} {}'.format(classname, "(Truth)")
            color = self.color_true[result]

            # Plot the boxes
            self.plot_box(draw, box, label, color)

    def plot_box(self, draw, box, label, color):
        """
        Actual routine for plotting the bounding boxes and labels on a PIL images
        :param draw: PIL.ImageDraw.draw object
        :param box: numpy array [x1, y1, x2, y2]
        :param label: string to include in the label
        :param color: color of the bounding box and label
        :return: Nothing
        """
        label_size = draw.textsize(label, self.font)

        # Draw bounding box
        draw.rectangle([(box[0], box[1]), (box[2], box[3])],
                       outline=color)
        # Draw label
        if box[1] - label_size[1] >= 0:
            text_origin = np.array([box[0], box[1] - label_size[1]])
        else:
            text_origin = np.array([box[0], box[1] + 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=self.font)


def COCO2KITTI(classes):
    """
    Converts from COCO class integers to KITTI class integers
    :param classes:
    :return:
    """
    for c in classes:
        if c == 0:    # person       (COCO)
            c = 3      # pedestrian  (KITTI)
        elif c == 1:  # bicycle
            c = 5      # cyclist
        elif c == 7:  # truck
            c = 2      # truck
        elif c == 2:  # car
            c = 0      # car
        else:
            c = 7      # Misc
    return c


def iou(box1, box2):
    # Implement the intersection over union (IoU) between box1 and box2
    #
    # Arguments:
    # box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    # box2 -- second box, list object with coordinates (x1, y1, x2, y2)

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)
    print(inter_area)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    print(union_area)

    # compute the IoU
    iou = inter_area / float(union_area)

    return iou

if __name__ == '__main__':
    loadFiles()