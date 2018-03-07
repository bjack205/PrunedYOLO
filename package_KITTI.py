
import os
from PIL import Image
import h5py as h5
import numpy as np
import argparse
import time
from random import shuffle
from yad2k.models.keras_yolo import preprocess_true_boxes

parser = argparse.ArgumentParser(
    description="Package KITTI dataset as npz.")
parser.add_argument('-d', '--data_path',
                    help="Path to the the root KITTI data set",
                    default="/KITTI")
parser.add_argument('-n', '--number_of_images',
                    help="Number of images to include in the dataset",
                    default=0,
                    type=int)
parser.add_argument('-o', '--output_path',
                    help="Output path for the .npz file",
                    default="./data")
parser.add_argument('-s', '--shuffle',
                    help="Shuffle the entire training set",
                    action='store_true')
parser.add_argument('-c', '--camera',
                    help="Camera number to import images from (default 2)",
                    default=2,
                    type=int)
parser.add_argument('-i', '--image_size',
                    help="Resolution of output images (width, height)",
                    default=(1200, 370),
                    nargs=2)

KITTI_LABELS = {'type': 0,
                'truncated': 1,
                'occluded': 2,
                'alpha': 3,
                'bbox': slice(4, 8),
                'dimensions': slice(8, 11),
                'location': slice(11, 14),
                'rotation': 14,
                'score': 15}
KITTI_CLASSES = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4,
                 'Cyclist': 5, 'Tram': 6, 'Misc': 7, 'DontCare': 8}

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def count_files(dir):
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


class KittiData:
    def __init__(self, m=-1, output_path='~/Research/PrunedYOLO/data'):
        self.data_path = "/KITTI"
        self.camera = 2
        self.shuffle = True
        self.dev_split = 0.1
        self.save_images = False
        self.image_size = (640, 192)  # Must be divisible by 32
        self.image_data_size = (416, 416)
        self.output_path = os.path.expanduser(output_path)
        self.batch_size = 32
        self.ext = ".png"
        self.ny = 5
        self.classes = KITTI_CLASSES
        self.anchors = YOLO_ANCHORS
        self.print_info = True

        self.grid_size = self.compute_grid_size()
        self.data_loaded = False
        self.saved_data_path = os.path.join(self.output_path, "KITTI-train.npz")
        self.h5file_path = os.path.join(self.output_path, "KITTI.h5")

        # Get file paths
        training_path = os.path.join(self.data_path, 'training')
        self.image_path = os.path.join(training_path, 'image_' + str(self.camera))
        self.label_path = os.path.join(training_path, 'label_' + str(self.camera))

        # Get total counts
        m_total = count_files(self.image_path)
        m_total_labels = count_files(self.label_path)
        assert (m_total == m_total_labels)

        # Get filename extensions
        image_names = os.listdir(self.image_path)
        _, self.im_ext = os.path.splitext(image_names[0])
        label_names = os.listdir(self.label_path)
        _, self.label_ext = os.path.splitext(label_names[0])

        # Remove extensions (to ensure we read matching image and label files)
        file_names = [os.path.splitext(name)[0] for name in image_names]
        file_names.sort()

        # Reduce data size
        if m > 0:
            file_names = file_names[:m]
            m_total = m

        # Split into train and dev
        if self.dev_split < 1:
            self.m_dev = int(np.round(m_total * self.dev_split))
        else:
            self.m_dev = self.dev_split
        self.m_train = int(m_total - self.m_dev)
        self.m_total = m_total
        dev_names = file_names[:self.m_dev]
        train_names = file_names[self.m_dev:self.m_dev + self.m_train]
        self.partition = {'train': train_names, 'dev': dev_names}
        self.IDs_list = file_names

        # Both dictionaries containing fields 'images' and 'labels'
        self.raw_data = {}
        self.data = {}

        # Dictionary of image/label IDs returning the index for self.data
        self.IDs = {}

        # Detector masks
        self.detector_masks = []
        self.matching_true_boxes = []

    def convert_to_h5(self, overwrite=None):
        h5path = self.h5file_path
        if os.path.exists(h5path):
            if overwrite is None:
                overwrite = input("h5 exists, would you like to overwrite it? (y/n)")
                if overwrite.lower() == 'y' or overwrite == 'yes':
                    write = True
                else:
                    write = False
            else:
                write = overwrite
        else:
            write = True
        if write:
            self.load_data()

            self.__print("Writing h5 file")
            file = h5.File(h5path, mode='w')
            self.__write_to_h5(file, self.partition["train"], "train")
            self.__write_to_h5(file, self.partition["dev"], "dev")
            file.close()

    def check_h5(self):
        """
        Checks if a valid h5 data file containing the processed data exists
        :return:
        """
        return os.path.exists(self.h5file_path)

    def load_data(self, yad2k=False):
        """
        Highest level data interaction function. Will detect if data has already been loaded.
        If data has been saved to a .npz file it will load it; otherwise it will parse the raw data files
        :return:
        """
        if not self.data_loaded:
            if os.path.exists(self.saved_data_path):
                self.load_files(yad2k)
            else:
                self.read_files(yad2k)
        else:
            self.__print("Data already loaded")

    def load_files(self, yad2k=False):
        """
        Loads saved npz files with the raw data
        :return: Nothing
        """
        self.__print("Loading data...")
        data_path = self.saved_data_path
        data = np.load(data_path)
        self.raw_data['images'] = data['images']
        self.raw_data['labels'] = data['image_labels']
        self.data['images'], self.data['labels'] = self.__strip_data(data['images'], data['image_labels'], yad2k=yad2k)
        del data
        num_images = self.data['images'].shape[0]
        num_labels = self.data['labels'].shape[0]
        if not num_labels == self.m_total:
            if self.save_images and not num_images == self.m_total:
                self.__print("Number of loaded images (%d) does not match expected (%d)" % (num_images, self.m_total))
            self.__print("Number of loaded labels (%d) does not match expected (%d)" % (num_labels, self.m_total))
        self.__print("Loaded %d images and %d labels" % (num_images, num_labels))
        self.data_loaded = True

    def read_files(self, yad2k=False):
        self.__print("Reading files...")
        image_data, image_labels = self.__read_files_raw(self.IDs_list)
        self.raw_data['labels'] = image_labels
        self.raw_data['images'] = image_data
        self.data['images'], self.data['labels'] = self.__strip_data(image_data, image_labels, yad2k=yad2k)
        self.data_loaded = True

    def save_yad2k_data(self, name):
        if self.save_images:
            self.load_data()
            images, boxes = self.__strip_data(self.raw_data['images'], self.raw_data['labels'], yad2k=True)
            save_path = os.path.join(self.output_path, name)
            np.savez(save_path, boxes=boxes, images=images)
            self.__print("Saved yad2k data to " + save_path + ".npz")

    def __strip_data(self, X, Y, yad2k=False):
        X_new = np.array(X)
        for i, im in enumerate(Y):
            ID = os.path.splitext(im['file'])[0]
            self.IDs[ID] = i
        Y_new = [im['objects'] for im in Y]
        if not yad2k:
            Y_new = self.convert_boxes(Y_new)
            self.detector_masks, self.matching_true_boxes = self.get_detector_mask(Y_new, self.anchors)
        else:
            Y_new = np.array(Y_new)
        return X_new, Y_new

    def __write_to_h5(self, file, IDs, name):
        """
        Saves data to h5 file read to read into the generator and into the training function
        Saves the data to named group (i.e. 'train' or 'dev') to split the data
        :param file: h5py dataset to write to
        :param IDs: ID of the images to write
        :param name: name of the group to write to

        Writes the following data:
        images: (m, h, w, 3) ndarray of image data, scaled from 0 to 1, with [w,h] defined by image_data_size
        labels: (m, n_b, 5) ndarray of labels [x, y, w, h, class]
        :return:
        """
        group = file.create_group(name)
        n = len(IDs)
        print("Number of images in" + name + ": " + str(n))
        label_size = self.data['labels'].shape
        image_shape = (n, self.image_data_size[1], self.image_data_size[0], 3)
        label_shape = (n, label_size[1], label_size[2])
        masks_shape = (n, self.grid_size[1], self.grid_size[0], len(self.anchors), 1)
        boxes_shape = (n, self.grid_size[1], self.grid_size[0], len(self.anchors), label_size[2])

        ID_len = int(len(IDs[0]))
        group.create_dataset("images", image_shape, np.float)
        group.create_dataset("labels", label_shape, np.float32)
        group.create_dataset("ids", (n,), '>S' + str(ID_len))
        group.create_dataset("detector_masks", masks_shape, np.bool)
        group.create_dataset("matching_true_boxes", boxes_shape, np.float32)

        for i, id in enumerate(IDs):
            if (i + 1) % 100 == 0:
                self.__print("Finished %d / %d" % (i + 1, n))
            index = self.IDs[id]
            im = self.__read_image(id)
            group["images"][i, ...] = im
            group["labels"][i, ...] = self.data['labels'][index, :, :]
            group["ids"][i] = np.string_(id)
            group["detector_masks"][i, ...] = self.detector_masks[index, :, :, :, :]
            group["matching_true_boxes"][i, ...] = self.matching_true_boxes[index, :, :, :, :]
        # group["ids"] = IDs

    def get_num_boxes(self):
        if self.check_h5():
            file = h5.File(self.h5file_path, mode='r')
            labels = file['dev']['labels']
            return labels.shape[1]
        else:
            return None

    def read_image_from_disk(self, ind):
        """
        Reads an image from the disk, specified by the index
        :param ind:
        :return:
        """
        if type(ind) is np.str_:
            name = ind
        else:
            name = self.IDs_list[ind]
        im_name = os.path.join(self.image_path, name + self.ext)
        im = Image.open(im_name)
        im = im.resize(self.image_size, Image.BICUBIC)
        im = np.array(im, dtype=np.uint8)
        return im

    def read_h5_dev_image(self, ind):
        if self.check_h5():
            file = h5.File(self.h5file_path, mode='r')
            images = file['dev']['images']
            im = images[ind, ...]
            return im
        else:
            return None

    def __read_image(self, id):
        """
        Reads an image from disk
        :param id: ID of the image to read from disk
        :return: ndarray of (h, w, 3) scaled from 0 to 1
        """
        im = Image.open(os.path.join(self.image_path, id + self.ext))
        im = np.array(im.resize(self.image_data_size, Image.BICUBIC), dtype=np.float)
        im = im / 255.0
        return im

    def __read_files_raw(self, file_names):
        """
        Parses through the raw KITTI data in self.data_path
        Reads both the images and the labels
        Only saves images when self.save_images is True
        Saves output to self.saved_data_path
        Saves "raw" data
        :param file_names: list of KITTI IDs to read
        :return:
            images: ndarray of image data, shape (m, width, height, channel)
            image_labels: (mx1) ndarray of objects. Each object is a dictionary:
                file: ID.png
                objects: ndarray of objects of size (num_objects x 5), columns [class, x1, y1, x2, y2]
                classes: list of class names of size (num_objects,)
        """
        # Initialize Lists
        image_labels = []
        images = []
        truncated = []
        occluded = []
        alpha = []
        dimensions = []
        location = []
        rotation_y = []
        score = []

        # Loop over the training examples
        for i, name in enumerate(file_names):

            im_name = os.path.join(self.image_path, name + self.im_ext)
            label_name = os.path.join(self.label_path, name + self.label_ext)
            if os.path.exists(im_name) and os.path.exists(label_name):
                # Read image
                im = Image.open(im_name)
                width = im.width
                height = im.height
                if self.save_images:
                    im = im.resize(self.image_size, Image.BICUBIC)
                    images.append(im)

                # Read label
                object_labels = []
                class_labels = []
                with open(label_name) as f:
                    label = f.readlines()
                for object in label:
                    # Parse the object line
                    object = object.split()
                    assert len(object) == 15

                    # Create label [type, x1, y1, x2, y2]
                    class_name = object[KITTI_LABELS['type']]
                    otype = KITTI_CLASSES[class_name]
                    bbox = [float(data) for data in object[KITTI_LABELS['bbox']]]
                    bbox[0] /= width / self.image_size[0]  # Rescale bounding boxes by target image size
                    bbox[1] /= height / self.image_size[1]
                    bbox[2] /= width / self.image_size[0]
                    bbox[3] /= height / self.image_size[1]
                    bbox.insert(0, otype)
                    object_labels.append(bbox)
                    class_labels.append(class_name)

                    # Store other KITTI data
                    truncated.append(float(object[KITTI_LABELS['truncated']]))
                    occluded.append(int(object[KITTI_LABELS['occluded']]))
                    alpha.append(float(object[KITTI_LABELS['alpha']]))
                    dimensions.append([float(data) for data in object[KITTI_LABELS['dimensions']]])
                    location.append([float(data) for data in object[KITTI_LABELS['location']]])
                    rotation_y.append(float(object[KITTI_LABELS['rotation']]))
                image_label = {'file': name + self.im_ext, 'objects': np.array(object_labels, dtype=np.int),
                               'classes': class_labels}
                image_labels.append(image_label)
            else:
                print("Something went wrong")
            if (i + 1) % 10 == 0:
                print("Finished %d images" % (i + 1))

        # Convert to numpy arrays
        image_labels = np.array(image_labels, dtype=np.object)
        images = [np.array(im) for im in images]
        images = np.array(images, dtype=np.uint8)

        # Output
        print("Dataset contains {} images".format(images.shape[0]))
        print("Dataset contains {} labels".format(image_labels.shape[0]))

        # Save dataset
        np.savez(self.saved_data_path,
                 filenames=file_names,
                 images=images,
                 image_labels=image_labels,
                 image_path=self.image_path,
                 truncated=truncated,
                 occluded=occluded,
                 alpha=alpha,
                 dimensions=dimensions,
                 location=location,
                 rotation=rotation_y,
                 image_size=self.image_size)
        print('Data saved: ' + self.saved_data_path)

        return images, image_labels

    def get_detector_mask(self, boxes, anchors):
        '''
        Precompute detectors_mask and matching_true_boxes for training.
        Detectors mask is 1 for each spatial position in the final conv layer and
        anchor that should be active for the given boxes and 0 otherwise.
        Matching true boxes gives the regression targets for the ground truth box
        that caused a detector to be active or 0 otherwise.
        Copied from YAD2K retrain_yolo.py
        '''
        detector_save_path = os.path.join(self.output_path, "KITTI-masks.npz")
        if os.path.exists(detector_save_path):
            self.__print("Loading detector masks from file...")
            data = np.load(detector_save_path)
            detectors_mask = data['detectors_mask']
            matching_true_boxes = data['matching_true_boxes']
        else:
            self.__print("Computing detector masks...")
            detectors_mask = [0 for i in range(len(boxes))]
            matching_true_boxes = [0 for i in range(len(boxes))]
            for i, box in enumerate(boxes):
                detectors_mask[i], matching_true_boxes[i] = \
                    preprocess_true_boxes(box, anchors, [self.image_data_size[1], self.image_data_size[0]])
            detectors_mask = np.array(detectors_mask)
            matching_true_boxes = np.array(matching_true_boxes)
            np.savez(detector_save_path, detectors_mask=detectors_mask, matching_true_boxes=matching_true_boxes)

        return np.array(detectors_mask, dtype=np.bool), np.array(matching_true_boxes)

    def convert_boxes(self, boxes):
        '''
        Converts the boxes to [x, y, w, h, class] for passing into the training algorithm
        :param boxes list of numpy arrays
        :return ndarray of (m, n_b, 5) with n_b equal to the maximum number of boxes (max 24)
                  boxes [x, y, w, h, class] in decimals
        '''
        self.__print("Converting boxes...")
        # Box preprocessing.
        orig_size = np.expand_dims(self.image_size, axis=0)
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]  # 397.5, 116.5
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]  # 51, 85
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros((max_boxes - boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(boxes)

    def compute_grid_size(self):
        return self.image_data_size[0] // 32, self.image_data_size[1] // 32

    def __print(self, string):
        if self.print_info:
            print(string)

    # GENERATOR FUNCTIONS
    def get_generators(self):
        if os.path.exists(self.h5file_path):
            self.__print("Created generators reading from " + self.h5file_path)
            train_generator = self.__generate('train')
            dev_generator = self.__generate('dev')
            return train_generator, dev_generator
        else:
            self.__print("No h5 file found")
            return [], []

    def __generate(self, name):
        'Generates batches of samples'
        # Infinite loop
        n = len(self.partition[name])
        batches_list = list(range(int(np.floor(float(n) / self.batch_size))))
        h5file = h5.File(self.h5file_path, mode='r')
        group = h5file[name]
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                shuffle(batches_list)

            # Generate batches
            if not batches_list:
                print("No batches!!!")
                yield None
            for j, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, n])  # index of the last image in this batch

                X, y = self.__data_generation(group, i_s, i_e)

                yield X, y


    def __data_generation(self, file, i_s, i_e):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        images = file["images"][i_s:i_e, ...]
        labels = file["labels"][i_s:i_e, ...]
        masks = file["detector_masks"][i_s:i_e, ...]
        boxes = file["matching_true_boxes"][i_s:i_e, ...]
        IDs = file["ids"][i_s:i_e, ...].astype(np.str)
        y = np.zeros((self.batch_size, 1))

        # num_anchors = len(self.anchors)
        # X = np.empty((self.batch_size, self.image_size[0], self.image_size[1], 3, 1))
        # boxes = np.empty((self.batch_size, self.data['labels'].shape[1], self.data['labels'].shape[2]))
        # detector_masks = np.empty((self.batch_size, self.grid_size[0], self.grid_size[1], num_anchors, 1))
        # true_boxes = np.empty((self.batch_size, self.grid_size[0], self.grid_size[1], num_anchors, 5))
        # y = np.zeros((self.batch_size, 1))
        #
        # # Generate data
        # for i, ID in enumerate(IDs):
        #     index = self.IDs[ID]
        #
        #     # Store inputs
        #     X[i, :, :, :, 0] = self.__read_image(ID)
        #     boxes[i, :, :] = self.data['labels'][index, :, :]
        #     detector_masks[i, :, :, :, :] = self.detector_masks[index, :, :, :, :]
        #     true_boxes[i, :, :, :, :] = self.matching_true_boxes[index, :, :, :, :]

        return [images, labels, masks, boxes, IDs], y



    @staticmethod
    def sparsify(y):
        'Returns labels in binary NumPy array'
        n_classes = 9  # Enter number of classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(y.shape[0])])


if __name__ == '__main__':
    # KE = KITTI_Extractor(parser.parse_args())
    # KE.extract()
    KD = KittiData(1000, "./data/medium")
    # KD.save_images = True
    # KD = KittiData()
    # KD.read_files()
    # KD.load_data()
    # KD.save_yad2k_data("KITTI_yad2k_tiny")
    # KD.shuffle = False
    # KD.load_files()
    # KD.convert_to_h5(overwrite=True)
    # KD.batch_size = 2

    print('\nTesting Generators')
    KD.batch_size = 1
    train_gen, dev_gen = KD.get_generators()
    print("Finished")

    if train_gen:

        tic = time.time()
        batch = next(train_gen)
        toc = time.time()-tic

        for out in batch[0]:
            print(out.shape)

        ID = batch[0][-1][0]
        print(ID + ".png")

        image = batch[0][0]
        assert np.isclose(np.max(image), 1)
        assert np.isclose(np.min(image), 0)
        print(toc)