import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf

import numpy as np
from package_KITTI import KittiData
from yad2k.models.keras_yolo import yolo_body, yolo_loss, yolo_eval, yolo_head, yolo_boxes_to_corners
from test_model import test

# from metrics import BoxPlotter
import argparse
import spar_keras
import spar

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

def prune_network():
    tic = time.time()
    print("Pruning...")
    sess = K.get_session()
    # writer = tf.summary.FileWriter("logs/prune/", sess.graph)

    num_pruned = spar.get_masks(sess, 0.01)
    spar.apply_masks(sess)
    print(str(time.time()-tic) + " seconds")
    return num_pruned

def count_params():
    sess = K.get_session()
    tf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    num_params = 0
    for var in tf_variables:
        var_params = np.prod(var.shape.as_list())
        num_params += var_params
    print("Current Parameters: ", num_params)

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    data = KittiData(m=1000, output_path="./data/coco", image_data_size=(608, 608), h5path="/KITTI/coco/KITTI.h5")
    # data = KittiData()
    num_classes = len(data.classes)
    model_body, model = create_model(data.image_data_size, data.anchors, data.classes, model_file="data/model_data/yolo.h5", freeze_body=False)

    # tic = time.time()
    # pl = spar_keras.get_prunable_layers(model_body)
    # cutoff = spar_keras.calc_cutoff(pl)
    # print(cutoff)
    # n_pruned = spar_keras.prune_weights(pl, cutoff)
    # print(time.time() - tic)
    # print(n_pruned)

    # tic = time.time()
    # prune_network(model_body)
    # print(time.time()-tic)

    train_with_pruning(data, weights_file="coco_retrain_full.h5")
    # train_gen(None, data, weights_file="coco_retrain_full.h5")
    # test(args, model_body, data, weights_file="coco_retrain_full.h5")

    # draw(model_body,
    #     class_names,
    #     anchors,
    #     image_data,
    #     image_set='val', # assumes training/validation split is 0.9
    #     weights_name='trained_stage_3_best.h5',
    #     save_all=False)



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
    count_params()
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
        data.batch_size = 16

        # Get data generators from H5 files
        train_gen, dev_gen = data.get_generators()

        # Compile the model
        model_body, model = create_model(data.image_data_size, data.anchors, data.classes,
                                         model_file="data/model_data/yolo.h5", freeze_body=False)
        # model_body, model = create_model(data.image_data_size, data.anchors, data.classes, load_pretrained=False, freeze_body=False)
        model.load_weights(weights_file)

        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        print("Starting training")
        run_name = "coco_pruning_full"
        model.fit_generator(generator=train_gen,
                            steps_per_epoch=len(data.partition['train']) // data.batch_size,
                            validation_data=dev_gen,
                            validation_steps=len(data.partition['dev']) // data.batch_size,
                            callbacks=[logging, checkpoint],
                            epochs=300
                            )
        model.save_weights(run_name + '.h5')

def train_with_pruning(data, weights_file):
    data.batch_size = 32
    num_epoch = 10

    # Get data generators from H5 files
    train_gen, dev_gen = data.get_generators()

    # Compile the model
    model_body, model = create_model(data.image_data_size, data.anchors, data.classes,
                                     model_file="data/model_data/yolo.h5", freeze_body=False)
    # model_body, model = create_model(data.image_data_size, data.anchors, data.classes, load_pretrained=False, freeze_body=False)
    model.load_weights(weights_file)

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    print("Starting training")
    num_batches = data.m_train // data.batch_size
    num_batches_dev = data.m_dev // data.batch_size
    min_loss = np.inf
    percent_pruned = []
    run_name = "coco_pruning_full"
    pruning_levels = [2, 4, 6, 8, 10]
    for level in pruning_levels:
        train_loss_avg = np.zeros((num_epoch,))
        dev_loss_avg = np.zeros((num_epoch,))

        for epoch in range(num_epoch):
            train_loss = np.zeros((num_batches,))
            dev_loss = np.zeros((num_batches_dev,))
            tic = time.time()
            pruned_weights = []

            # Re-calculate pruning each epoch
            sess = K.get_session()
            percent_pruned.append(spar.get_masks(sess, level))

            for i in range(num_batches):
                tic = time.time()
                x, y = next(train_gen)
                train_loss[i] = model.train_on_batch(x, y)
                spar.apply_masks(sess)  # Re-apply masks to cancel gradient update on pruned weights
                print("Epoch %d: batch %d / %d - Loss: %.2f (%.2f seconds)" % (
                epoch + 1, i + 1, num_batches, train_loss[i], time.time() - tic))
            for j in range(num_batches_dev):
                x, y = next(dev_gen)
                dev_loss[j] = model.test_on_batch(x, y)
            train_loss_avg[epoch] = float(np.mean(train_loss))
            dev_loss_avg[epoch] = float(np.mean(dev_loss))
            toc = time.time() - tic
            print("\nEpoch %d: Train Loss = %.2f, Dev Loss = %.2f (%.1f sec)"
                  % (epoch, train_loss_avg[epoch], dev_loss_avg[epoch], toc))
            model.save_weights(run_name + "_epoch_" + str(epoch) + "_pruning_" + str(level) + ".h5")

        np.savez(run_name + "_prune_" + str(level) + "_stats",
                 weights=percent_pruned,
                 train_loss=train_loss_avg,
                 dev_loss=dev_loss_avg)

    model.save_weights(run_name + '.h5')



if __name__=='__main__':
    args = argparser.parse_args()
    _main(args)