from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf

import numpy as np
from package_KITTI import KittiData
from yad2k.models.keras_yolo import yolo_body, yolo_loss, yolo_eval, yolo_head, yolo_boxes_to_corners


def _main():
    data = KittiData(m=1000, output_path="./data/coco", image_data_size=(608, 608), h5path="/KITTI/coco/KITTI.h5")



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
    _main()