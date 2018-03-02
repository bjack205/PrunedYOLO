from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os
import argparse
import numpy as np
from package_KITTI import KittiData
from YAD2K.yad2k.models.keras_yolo import yolo_body, yolo_loss
import tensorflow as tf

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



def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    data = KittiData()
    num_classes = len(data.classes)
    model_body, model = create_model(data.image_size, data.anchors, data.classes)

    train_gen(
        model,
        data
    )
    #
    # draw(model_body,
    #     class_names,
    #     anchors,
    #     image_data,
    #     image_set='val', # assumes training/validation split is 0.9
    #     weights_name='trained_stage_3_best.h5',
    #     save_all=False)


def create_model(image_size, anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (image_size[1]//32, image_size[0]//32, len(anchors), 1)
    matching_boxes_shape = (image_size[1]//32, image_size[0]//32, len(anchors), 5)

    # Create model input layers.
    image_input = Input(shape=(image_size[1], image_size[0], 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

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
         matching_boxes_input], model_loss)

    return model_body, model


def train_gen(model, data):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''

    # Fine Tuning
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    data.batch_size = 32
    train_gen, dev_gen = data.get_generators()
    logging = TensorBoard()
    batch = next(train_gen)
    for out in batch[0]:
        print(out.shape)
    model.load_weights("../trained_stage_2.h5")
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=len(data.partition['train'])//data.batch_size,
                        validation_data=dev_gen,
                        validation_steps=len(data.partition['dev'])//data.batch_size,
                        callbacks=[logging],
                        epochs=5
                        )
    model.save_weights('fine_tuning.h5')

    # Stage II
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)

    model_body, model = create_model(data.anchors, len(data.classes), load_pretrained=False, freeze_body=False)
    model.load_weights('fine_tuning.h5')
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

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