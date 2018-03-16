import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf

import numpy as np
from package_KITTI import KittiData
from yad2k.models.keras_yolo import yolo_body, yolo_loss, yolo_eval, yolo_head, yolo_boxes_to_corners
from test_model import test

# from metrics import BoxPlotter
from create_model import create_model
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

    data = KittiData(m=100, output_path="./data/tiny", image_data_size=(608, 608), h5path="data/tiny/KITTI.h5")
    # data = KittiData()
    num_classes = len(data.classes)
    # model_body, model = create_model(data.image_data_size, data.anchors, data.classes, model_file="data/model_data/yolo.h5", freeze_body=False)

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

    # train_with_pruning(data, weights_file="coco_retrain_full.h5")
    train_gen(None, data, weights_file="coco_retrain_full.h5")
    test(args, model_body, data, weights_file="trained_checkpoint_best.h5")

    # draw(model_body,
    #     class_names,
    #     anchors,
    #     image_data,
    #     image_set='val', # assumes training/validation split is 0.9
    #     weights_name='trained_stage_3_best.h5',
    #     save_all=False)


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
        data.batch_size = 1

        # Get data generators from H5 files
        train_gen, dev_gen = data.get_generators()

        # Compile the model
        model_body, model = create_model(data.image_data_size, data.anchors, data.classes,
                                         model_file="data/model_data/yolo.h5", freeze_body=False)
        prunable_layers = spar_keras.get_prunable_layers(model)
        cutoff = spar_keras.calc_cutoff(prunable_layers)
        pruning = PruneCallback(cutoff)

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
                            callbacks=[logging, checkpoint, pruning],
                            epochs=300
                            )
        model.save_weights(run_name + '.h5')

class PruneCallback(Callback):
    def __init__(self, CUTOFF):
        super(PruneCallback, self).__init__()
        self.CUTOFF = CUTOFF

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
         return

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