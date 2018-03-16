import time
from keras import backend as K

import numpy as np
from package_KITTI import KittiData
from yad2k.models.keras_yolo import yolo_body, yolo_loss, yolo_eval, yolo_head, yolo_boxes_to_corners
from metrics import CalcMetrics


def test(args, model, data, weights_file=None):
    model_path = "data/model_data/yolo.h5"
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    output_path = data.output_path
    score_threshold = 0.1
    iou_threshold = 0.5

    class_names = data.classes
    anchors = data.anchors

    # Plotter
    plot = False
    # plotter = BoxPlotter(data.image_size, data.classes)

    # yolo_model = load_model(model_path)
    model.load_weights(weights_file)
    yolo_model = model

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    output_shape = yolo_model.layers[-1].output_shape
    model_output_channels = output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = model.layers[0].input_shape[1:3]
    print("input shape ", model_image_size)


    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    max_boxes = data.get_num_boxes()
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        max_boxes=max_boxes,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)
    # print("boxes: ", boxes.shape, "\nscores: ", scores.shape, "\nclasses: ", classes.shape)


    # Set up variables
    data.batch_size = 1   # Can't do batches because yolo_eval doesn't support it
    data.shuffle = False
    t_gen, val_gen = data.get_generators()
    print(yolo_model.input.shape)
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    m = data.m_dev
    boxes_true = []
    scores_true = np.ones((m, max_boxes))
    classes_true = np.zeros((m, max_boxes))
    IDs = data.partition['dev']
    boxes_pred = []
    scores_pred = []
    classes_pred = []

    print("max boxes:", max_boxes)

    tic = time.time()
    # Loop over each image and test
    IDs = []
    for i in range(m):
        # Get next training sample
        val = next(val_gen)

        # Extract information from sample
        image = val[0][0]  # decimal values from 0 to 1
        label_true = val[0][1][0, ...]  # [x, y, w, h, class]
        box_xy = label_true[:, 0:2]
        box_wh = label_true[:, 2:4]
        class_num = label_true[:, -1].astype(np.int)
        box = label_to_box(box_xy, box_wh, data.image_size)
        box = box[:, [1, 0, 3, 2]]
        ID = val[0][-1][0]

        boxes_true.append(box)
        IDs.append(ID)
        classes_true[i, :] = class_num
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image,
                input_image_shape: [data.image_size[1], data.image_size[0]],
                K.learning_phase(): 0
            })
        # outputs boxes [y2, x2, y1, x2]
        out_boxes = out_boxes[:, [1, 0, 3, 2]]
        print(str(out_boxes.shape[0]) + " objects detected")

        boxes_pred.append(pad_objects(out_boxes, max_boxes))
        scores_pred.append(pad_objects(out_scores, max_boxes))
        classes_pred.append(pad_objects(out_classes, max_boxes))
        # print("Out Shapes\nBoxes: ", out_boxes.shape, "\nscores: ", out_scores.shape, "\nclasses: ", out_classes.shape)

        # full_image = data.read_image_from_disk(ID)
        # yhat = plotter.package_data(out_boxes, out_classes, out_scores)
        # y = plotter.package_data(box, class_num, ID)
        # plotter.comparison(y, yhat, full_image)
        # input("Enter for next image")
        if (i+1) % 25 == 0:
            print("Finished Predictions for %d / %d" % (i+1, m))
    toc = time.time() - tic
    print(toc)
    # print("Predictions per second: %f.2" % toc/data.m_dev)

    boxes_true = np.array(boxes_true)
    boxes_pred = np.array(boxes_pred)
    classes_pred = np.array(classes_pred)
    scores_pred = np.array(scores_pred)
    IDs = np.array(IDs)
    assert boxes_true.shape == boxes_pred.shape
    assert classes_true.shape == classes_pred.shape
    assert scores_true.shape == scores_pred.shape

    y = {'boxes': boxes_true, 'classes': classes_true, 'scores': scores_true, 'ID': IDs}
    yhat = {'boxes': boxes_pred, 'classes': classes_pred, 'scores': scores_pred, 'ID': IDs}

    np.savez("Predictions.npz", y=y, yhat=yhat)
    CalcMetrics(y, yhat, data=data)

    sess.close()

def pad_objects(X, max_objects):
    num_objects = X.shape[0]
    pad = np.zeros((max_objects - num_objects, *X.shape[1:]))
    X = np.concatenate((X, pad), axis=0)
    return X

def label_to_box(box_xy, box_wh, image_size):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return np.hstack([
        box_mins[:, 1:2]*image_size[1],  # y_min
        box_mins[:, 0:1]*image_size[0],  # x_min
        box_maxes[:, 1:2]*image_size[1],  # y_max
        box_maxes[:, 0:1]*image_size[0]  # x_max
    ])