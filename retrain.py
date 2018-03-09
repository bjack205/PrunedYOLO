import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import argparse
import numpy as np
from package_KITTI import KittiData
from yad2k.models.keras_yolo import yolo_body, yolo_loss, yolo_eval, yolo_head, yolo_boxes_to_corners
import tensorflow as tf
from metrics import BoxPlotter
import spar_v3

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default=os.path.join('..', 'DATA', 'underwater_data.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))


def prune_network(model_body):
    sess = K.get_session()
    writer = tf.summary.FileWriter("logs/prune/", sess.graph)

    spar_v3.get_masks(sess, 0.01)
    spar_v3.apply_masks(sess)

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    data = KittiData(m=1000, output_path="./data/coco", image_data_size=(608, 608))
    # data = KittiData()
    num_classes = len(data.classes)
    model_body, model = create_model(data.image_data_size, data.anchors, data.classes, model_file="data/model_data/yolo.h5")

    train_gen(model, data, weights_file="data/model_data/yolo_weights.h5")
    # test(args, model_body, data, weights_file="data/weights/fine_tuning_withgen_300e_1000images.h5")

    # draw(model_body,
    #     class_names,
    #     anchors,
    #     image_data,
    #     image_set='val', # assumes training/validation split is 0.9
    #     weights_name='trained_stage_3_best.h5',
    #     save_all=False)

def test(args, model, data, weights_file=None):
    model_path = "data/model_data/yolo.h5"
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    output_path = data.output_path
    score_threshold = 0.7
    iou_threshold = 0.5

    class_names = data.classes
    anchors = data.anchors

    # Plotter
    plotter = BoxPlotter(data.image_size, data.classes)

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
        # print(str(out_boxes.shape[0]) + " objects detected")

        boxes_pred.append(pad_objects(out_boxes, max_boxes))
        scores_pred.append(pad_objects(out_scores, max_boxes))
        classes_pred.append(pad_objects(out_classes, max_boxes))
        # print("Out Shapes\nBoxes: ", out_boxes.shape, "\nscores: ", out_scores.shape, "\nclasses: ", out_classes.shape)

        full_image = data.read_image_from_disk(ID)
        yhat = plotter.package_data(out_boxes, out_classes, out_scores)
        y = plotter.package_data(box, class_num, ID)
        # plotter.comparison(y, yhat, full_image)
        # input("Enter for next image")
        if (i+1) % 100 == 0:
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

def create_model(image_size, anchors, class_names, model_file=None, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''
    if not model_file is None:
        print("USING MODEL FILE")
        model_body = load_model(model_file)
        input_shape = model_body.layers[0].input_shape
        image_input_size = [input_shape[2], input_shape[1]]
        print(image_input_size)
        print(image_size)
        assert np.allclose(image_input_size, image_size)

    detectors_mask_shape = (image_size[1]//32, image_size[0]//32, len(anchors), 1)
    matching_boxes_shape = (image_size[1]//32, image_size[0]//32, len(anchors), 5)

    # Create model input layers.
    image_input = Input(shape=(image_size[1], image_size[0], 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)
    ids_input = Input(shape=(1,))

    if model_file is None:
        # Create model body.
        yolo_model = yolo_body(image_input, len(anchors), len(class_names))

        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        if load_pretrained:
            # Save topless yolo:
            topless_yolo_path = os.path.join('data/model_data', 'yolo_topless.h5')
            if not os.path.exists(topless_yolo_path):
                print("CREATING TOPLESS WEIGHTS FILE")
                yolo_path = os.path.join('data/model_data', 'yolo.h5')
                model_body = load_model(yolo_path)
                model_body = Model(model_body.inputs, model_body.layers[-2].output)
                model_body.save_weights(topless_yolo_path)
            topless_yolo.load_weights(topless_yolo_path)

        final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

        model_body = Model(image_input, final_layer)

    if freeze_body:
        for layer in model_body.layers[:-1]:
            layer.trainable = False

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input, ids_input], model_loss)

    return model_body, model


def train_gen(model, data, weights_file="YOLO_fine_tuned.h5"):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    logging = TensorBoard()
    checkpoint_tuning = ModelCheckpoint("fine_tuning_checkpoint.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    checkpoint = ModelCheckpoint("trained_checkpoint_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    if 0:
        # model.load_weights('fine_tuning.h5')

        # Fine Tuning
        model.load_weights(weights_file)
        prune_network(model)

        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        data.batch_size = 4
        train_gen, dev_gen = data.get_generators()

        batch = next(train_gen)
        for out in batch[0]:
            print(out.shape)
        model.fit_generator(generator=train_gen,
                            steps_per_epoch=len(data.partition['train'])//data.batch_size,
                            validation_data=dev_gen,
                            validation_steps=len(data.partition['dev'])//data.batch_size,
                            callbacks=[logging, checkpoint_tuning],
                            epochs=400
                            )
        weight_save_file = "coco_fine_tuning_withgen.h5"
        print("Finished Fine-tuning, saving weight as " + weight_save_file)
        model.save_weights(weight_save_file)


    # Stage II
    if 1:
        data.batch_size = 1
        train_gen, dev_gen = data.get_generators()

        model_body, model = create_model(data.image_data_size, data.anchors, data.classes, load_pretrained=False, freeze_body=False)
        model.load_weights(weights_file)

        prune_network(model)
        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        num_epoch = 10
        num_batches = data.m_train // data.batch_size
        num_batches_dev = data.m_dev // data.batch_size
        min_loss = np.inf
        for epoch in range(num_epoch):
            train_loss = np.zeros((num_batches,))
            dev_loss = np.zeros((num_batches_dev,))
            tic = time.time()
            for i in range(num_batches):
                x, y = next(train_gen)
                train_loss[i] = model.train_on_batch(x, y)
                prune_network(model)
            for j in range(num_batches_dev):
                x, y = next(dev_gen)
                dev_loss[j] = model.test_on_batch(x, y)
            train_loss_avg = float(np.mean(train_loss))
            dev_loss_avg = float(np.mean(dev_loss))
            toc = time.time() - tic
            print("Epoch %d: Train Loss = %f.2, Dev Loss = %f.2 (%f.1 sec)"
                  % (epoch, train_loss_avg, dev_loss_avg, toc))

            

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=len(data.partition['train']) // data.batch_size,
                            validation_data=dev_gen,
                            validation_steps=len(data.partition['dev']) // data.batch_size,
                            callbacks=[logging, checkpoint],
                            epochs=30
                            )
        model.save_weights('KITTI_retrain_full.h5')



if __name__=='__main__':
    args = argparser.parse_args()
    _main(args)