
import os
from PIL import Image
import h5py as h5
import numpy as np
import argparse
import time
from random import shuffle
from YAD2K.yad2k.models.keras_yolo import preprocess_true_boxes

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
                    default=".")
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
    def __init__(self):
        self.data_path = "/KITTI"
        self.camera = 2
        self.shuffle = True
        self.dev_split = 0.1
        self.save_images = False
        self.image_size = (1248, 384)  # Must be divisible by 32
        self.output_path = '.'
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

        # Split into train and dev
        if self.dev_split < 1:
            self.m_dev = int(np.round(m_total * self.dev_split))
        else:
            self.m_dev = self.dev_split
        self.m_train = int(m_total - self.m_dev)
        dev_names = file_names[:self.m_dev]
        train_names = file_names[self.m_dev:self.m_dev + self.m_train]
        self.partition = {'train': train_names, 'dev': dev_names}
        self.IDs = file_names

        # Both dictionaries containing fields 'images' and 'labels'
        self.raw_data = {}
        self.data = {}

        # Dictionary of image/label IDs returning the index for self.data
        self.IDs = {}

        # Detector masks
        self.detector_masks = []
        self.matching_true_boxes = []

    def compute_grid_size(self):
        return self.image_size[0] // 32, self.image_size[1] // 32

    def __print(self, string):
        if self.print_info:
            print(string)

    def convert_to_h5(self):
        h5path = self.h5file_path
        if os.path.exists(h5path):
            overwrite = input("h5 exists, would you like to overwrite it? (y/n)")
            if overwrite.lower() == 'y' or overwrite == 'yes':
                write = True
            else:
                write = False
        else:
            write = True
        if write:
            self.load_data()

            self.__print("Writing h5 file")
            file = h5.File(h5path, mode='w')
            self.__write_to_h5(file, self.partition["train"], "train")
            self.__write_to_h5(file, self.partition["dev"], "dev")
            file.close()

    def __write_to_h5(self, file, IDs, name):
        group = file.create_group(name)
        n = len(IDs)
        label_size = self.data['labels'].shape
        image_shape = (n, self.image_size[0], self.image_size[1], 3)
        label_shape = (n, label_size[1], label_size[2])
        masks_shape = (n, self.grid_size[0], self.grid_size[1], len(self.anchors), 1)
        boxes_shape = (n, self.grid_size[0], self.grid_size[1], len(self.anchors), label_size[2])

        ID_len = int(len(IDs[0]))
        group.create_dataset("images", image_shape, np.int8)
        group.create_dataset("labels", label_shape, np.float32)
        group.create_dataset("ids", (n,), '>S' + str(ID_len))
        group.create_dataset("detector_masks", masks_shape, np.bool)
        group.create_dataset("matching_true_boxes", boxes_shape, np.float32)

        for i, id in enumerate(IDs):
            if (i + 1) % 100 == 0:
                self.__print("Finished %d / %d" % (i+1, n))
            index = self.IDs[id]
            im = self.__read_image(id)
            group["images"][i, ...] = im
            group["labels"][i, ...] = self.data['labels'][index, :, :]
            group["ids"][i] = np.string_(id)
            group["detector_masks"][i, ...] = self.detector_masks[index, :, :, :, :]
            group["matching_true_boxes"][i, ...] = self.matching_true_boxes[index, :, :, :, :]
        # group["ids"] = IDs


    def __read_image(self, id):
        im = Image.open(os.path.join(self.image_path, id + self.ext))
        im = np.array(im.resize(self.image_size, Image.BICUBIC)).transpose((1, 0, 2))
        return im

    def load_data(self):
        if not self.data_loaded:
            if os.path.exists(self.saved_data_path):
                self.load_files()
            else:
                self.read_files()
        else:
            self.__print("Data already loaded")


    def load_files(self):
        """
        Loads saved npz files with the raw data
        :return: Nothing
        """
        self.__print("Loading data...")
        data_path = self.saved_data_path
        data = np.load(data_path)
        self.raw_data['images'] = data['images']
        self.raw_data['labels'] = data['image_labels']
        self.data['images'], self.data['labels'] = self.__strip_data(data['images'], data['image_labels'])
        del data
        self.data_loaded = True

    def read_files(self):
        self.__print("Reading files...")
        image_data, image_labels = self.__read_files_raw(self.IDs)
        self.raw_data['labels'] = image_labels
        self.raw_data['images'] = image_data
        self.data['images'], self.data['labels'] = self.__strip_data(image_data, image_labels)
        self.data_loaded = True

    def __strip_data(self, X, Y):
        X_new = np.array(X)
        for i, im in enumerate(Y):
            ID = os.path.splitext(im['file'])[0]
            self.IDs[ID] = i
        Y_new = [im['objects'] for im in Y]
        Y_new = self.convert_boxes(Y_new)
        self.detector_masks, self.matching_true_boxes = self.get_detector_mask(Y_new, self.anchors)
        return X_new, Y_new

    def convert_boxes(self, boxes):
        '''processes the data'''
        self.__print("Converting boxes...")
        # Box preprocessing.
        orig_size = np.expand_dims(self.image_size, axis=0)
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
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
                detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, self.image_size)
            detectors_mask = np.array(detectors_mask)
            matching_true_boxes = np.array(matching_true_boxes)
            np.savez(detector_save_path, detectors_mask=detectors_mask, matching_true_boxes=matching_true_boxes)

        return np.array(detectors_mask, dtype=np.bool), np.array(matching_true_boxes)

    def __read_files_raw(self, file_names):
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
                    images.append(im.resize(self.image_size))

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
        images = [np.array(im) / 255. for im in images]
        images = np.array(images, dtype=np.uint8)

        # Output
        print("Dataset contains {} images".format(images.shape[0]))
        print("Dataset contains {} labels".format(image_labels.shape[0]))

        # Save dataset
        np.savez(self.data_path,
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
        print('Data saved: ' + self.data_path)

        return images, image_labels

    # GENERATOR FUNCTIONS
    def get_generators(self):
        train_generator = self.__generate('train')
        dev_generator = self.__generate('dev')
        return train_generator, dev_generator

    def __generate(self, name):
        'Generates batches of samples'
        # Infinite loop
        n = len(self.partition[name])
        batches_list = list(range(int(np.ceil(float(n) / self.batch_size))))
        h5file = h5.File(self.h5file_path, mode='r')
        group = h5file[name]
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                shuffle(batches_list)

            # Generate batches
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

        return [images, labels, masks, boxes], y

    @staticmethod
    def sparsify(y):
        'Returns labels in binary NumPy array'
        n_classes = 9  # Enter number of classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(y.shape[0])])


if __name__ == '__main__':
    # KE = KITTI_Extractor(parser.parse_args())
    # KE.extract()
    KD = KittiData()
    # KD.read_files()
    # KD.load_files()
    # KD.convert_to_h5()

    print('\nTesting Generators')
    train_gen, dev_gen = KD.get_generators()
    batch = next(train_gen)

    tic = time.time()
    batch = next(train_gen)
    toc = time.time()-tic
    print(toc)