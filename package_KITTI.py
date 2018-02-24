
import os
import PIL.Image
import numpy as np
import argparse

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
                    help="Output path for the .npz file")
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


def count_files(dir):
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


def _main(args):
    data_path = args.data_path
    camera = args.camera
    shuffle = args.shuffle
    output_path = args.output_path
    image_size = args.image_size

    # Get file paths
    training_path = os.path.join(data_path, 'training')
    image_path = os.path.join(training_path, 'image_' + str(camera))
    label_path = os.path.join(training_path, 'label_' + str(camera))

    # Get total counts
    m_train = count_files(image_path)
    m_train_labels = count_files(label_path)
    if args.number_of_images > 0:
        images = args.number_of_images
    else:
        images = 100 # m_train
    assert (m_train == m_train_labels)

    # Get filename extensions
    image_names = os.listdir(image_path)
    _, im_ext = os.path.splitext(image_names[0])
    label_names = os.listdir(label_path)
    _, label_ext = os.path.splitext(label_names[0])

    # Remove extensions (to ensure we read matching image and label files)
    train_names = [os.path.splitext(name)[0] for name in image_names]
    train_names = train_names[:images]



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
    for i, name in enumerate(train_names):

        im_name = os.path.join(image_path, name + im_ext)
        label_name = os.path.join(label_path, name + label_ext)
        if os.path.exists(im_name) and os.path.exists(label_name):
            # Read image
            im = PIL.Image.open(im_name)
            width = im.width
            height = im.height
            images.append(im.resize(image_size))

            # Read label
            image_label = []
            with open(label_name) as f:
                label = f.readlines()
            for object in label:
                # Parse the object line
                object = object.split()
                assert len(object) == 15

                # Create label [type, x1, y1, x2, y2]
                otype = KITTI_CLASSES[object[KITTI_LABELS['type']]]
                bbox = [float(data) for data in object[KITTI_LABELS['bbox']]]
                bbox[0] /= width*image_size[0]  # Rescale bounding boxes by target image size
                bbox[1] /= height*image_size[1]
                bbox[2] /= width*image_size[0]
                bbox[3] /= height*image_size[1]
                bbox.insert(0, otype)
                image_label.append(bbox)

                # Store other KITTI data
                truncated.append(float(object[KITTI_LABELS['truncated']]))
                occluded.append(int(object[KITTI_LABELS['occluded']]))
                alpha.append(float(object[KITTI_LABELS['alpha']]))
                dimensions.append([float(data) for data in object[KITTI_LABELS['dimensions']]])
                location.append([float(data) for data in object[KITTI_LABELS['location']]])
                rotation_y.append(float(object[KITTI_LABELS['rotation']]))
            image_labels.append(np.array(image_label,  dtype=np.int32))
        else:
            print("Something went wrong")
        if i % 10 == 0:
            print("Finished %d images" % i)
    # Convert to numpy arrays
    image_labels = np.array(image_labels, dtype=np.object)
    images = [np.array(im) for im in images]
    images = np.array(images, dtype=np.uint8)

    # Output
    print("Dataset contains {} images".format(images.shape[0]))

    # Save dataset
    np.savez("my_dataset",
             images=images,
             boxes=image_labels,
             truncated=truncated,
             occluded=occluded,
             alpha=alpha,
             dimensions=dimensions,
             location=location,
             rotation=rotation_y)
    print('Data saved: my_dataset.npz')

    return images, image_labels

if __name__ == '__main__':
    _main(parser.parse_args())
