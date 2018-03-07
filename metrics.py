import numpy as np
import package_KITTI
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageColor, ImageFont
from prettytable import PrettyTable

KITTI_CLASSES_LIST = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                     'Cyclist', 'Tram', 'Misc', 'DontCare']

KITTI_CLASSES = package_KITTI.KITTI_CLASSES

def loadFiles():
    predictions_file = "Predictions.npz"
    data = np.load(predictions_file)
    y = data['y'].item()
    yhat = data['yhat'].item()
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
    data = package_KITTI.KittiData(m=1000, output_path="data/medium")

    # Parameters
    iou_threshold = 0.5      # Threshold for an accurate localization
    plot = True              # Plot the images with bounding boxes
    car_forgiveness = True   # Allows DontCare true labels to be accepted as cars
    num_images = 900         # Number of images to analyze

    # Important vars
    num_classes = len(data.classes)

    # Get image size
    imsize_true = data.image_size

    # Instatiate the plotter object
    plotter = BoxPlotter(data.image_size)
    plotter.save_path = "output_images/"

    # Set up variables
    partition = 'dev'
    if partition == 'dev':
        m = data.m_dev
    else:
        m = data.m_train
    m = np.minimum(m, num_images)
    TP = np.zeros(num_classes)  # True positive (corrected)
    FP = np.zeros(num_classes)  # False positive (detected but incorrect)
    FN = np.zeros(num_classes)  # False negative (not detected)
    CN = np.zeros(num_classes)  # Count of true labels

    # Loop over all of the images
    for i in range(m):
        ID = y['ID'][i]
        image = data.read_image_from_disk(ID)

        # Extract true label data
        bb_true = y['boxes'][i, ...]
        objects = ~np.all(bb_true == 0, axis=-1)
        bb_true = bb_true[objects, ...]
        c_true = y['classes'][i, objects].astype(int)
        num_objects = bb_true.shape[0]

        # Extract predicted label data
        boxes_pred = yhat['boxes'][i, ...]
        objects = ~np.all(boxes_pred == 0, axis=-1)
        boxes_pred = boxes_pred[objects, ...]
        classes_pred = yhat['classes'][i, objects, ...].astype(np.int)
        scores = yhat['scores'][i, objects, ...]

        num_predictions = boxes_pred.shape[0]

        matches = np.zeros(num_objects)
        result = np.zeros(num_predictions)  # track if the box was FP (0) or TP (1)

        # Loop over each predicted detection
        for j, bbhat in enumerate(boxes_pred):
            chat = int(classes_pred[j])

            # Match the bounding box with the bounding box with the greatest IOU
            ious = [iou(bbhat, bb) for bb in bb_true]
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
        # yhat[i]['result'] = result

        # Count all true boxes that were never matched as false negatives
        # y[i]['result'] = matches > 0
        for j in (matches == 0).nonzero():
            FN[c_true[j]] += 1

        # Get true counts for stats
        for c in c_true:
            CN[c] += 1

        # if np.any(matches > 1):
        #     print("Some boxes were detected more than once")

        # Plot
        if plot:
            y_i = plotter.package_data(bb_true, c_true, ID)
            yhat_i = plotter.package_data(boxes_pred, classes_pred, ID, scores, result)
            # yhat_i = {'boxes': boxes_pred, 'classes': classes_pred, 'scores': scores, 'result': result}
            plotter.comparison(y_i, yhat_i, image)

        if (i+1) % 100 == 0:
            print("Finished %d / %d Images" % (i+1, m))

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
    def __init__(self, image_size, save_path=None):
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
        self.save_path = save_path

    def package_data(self, boxes, classes, ID=None, scores=None, results=None):
        y = {"boxes": boxes, "classes": classes}
        if not ID is None:
            y["ID"] = ID
        if not scores is None:
            y["scores"] = scores
        if not results is None:
            y["results"] = results
        return y

    def comparison(self, y, yhat, image_data):
        """
        Plots both predicted and true bounding boxes for comparison
        :param y: dictionary of a true label data. One entry of the "y" input to the CalcMetrics function
        :param yhat: dictionary of predicted label data
        :return: Nothing. Displays a plot to the screen
        """
        # Open image and set up drawing variables
        image_data = image_data / np.max(image_data) * 255
        image_data = image_data.astype(np.uint8)
        ID = y['ID']
        # plt.imshow(image_data)
        # plt.show()


        image = Image.fromarray(image_data)
        draw = ImageDraw.Draw(image)

        # Plot the boxes
        self.truth_boxes(draw, y)
        self.prediction_boxes(draw, yhat)
        # image.show()

        if not self.save_path is None:
            image.save(os.path.join(os.path.join(self.save_path, ID + ".png")))

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
        num_detections = yhat['scores'].shape[0]
        for j in range(num_detections):
            # Extract out import info from dictionary
            classname = KITTI_CLASSES_LIST[yhat['classes'][j]]
            score = yhat['scores'][j]
            result = 1  # int(yhat['result'][j])
            box = yhat['boxes'][j, :]

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
        num_true = y['classes'].shape[0]
        for j in range(num_true):
            # Extract important info from dictionary
            classname = KITTI_CLASSES_LIST[y['classes'][j]]
            result = 1
            box = y['boxes'][j, :]
            # box = box[[1, 0, 3, 2]]

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

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / float(union_area)

    return iou

if __name__ == '__main__':
    loadFiles()