#This is the library for weights sparsification

import tensorflow as tf
import numpy as np

# masks is a list of tuple, each tuple is (var_name, mask)
masks = []

# name_tfv is a dictionary, where key is the variable name,
# and value is tf.variable
name_tfv = {}

# name_ph, key variable name and value is tf.placeholder (with same dtype and shape)
name_ph = {}


def get_masks(sess, percent):
    # get_mask generates masks sparsifying model weights
    # Criteria for sparsification is specified by percientile,
    # i.e. fraction of elements masked
    # arguments:
    # sess: tf sess where weights are stored
    # percent: fraction of sparsified
    tf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # name_vals is list, each is tuple (var_name, value)
    name_vals = []
    data = []

    # old_vals only contains values, it's a list
    old_vals = sess.run(tf_variables)
    for val in old_vals:
        if len(val.shape) > 1:
            data.extend(np.absolute(val).flatten().tolist())

    CUTOFF = np.percentile(data, percent)

    for i in range(len(tf_variables)):
        if len(old_vals[i].shape) > 1:
            name_tfv[tf_variables[i].name] = tf_variables[i]
            cur_ph = tf.placeholder(tf_variables[i].dtype, shape=tf_variables[i].get_shape())
            name_ph[tf_variables[i].name] = cur_ph
            name_vals.append((tf_variables[i].name, old_vals[i]))

    global masks
    for pair in name_vals:
        weight_val = pair[1]
        mask_cur = np.ones(weight_val.shape)
        if len(weight_val.shape) == 2:
            for i in range(weight_val.shape[0]):
                for j in range(weight_val.shape[1]):
                    if abs(weight_val[i, j]) < CUTOFF:
                        mask_cur[i, j] = 0
        #pair[0] is the variable name

        masks.append((pair[0], mask_cur))
    return masks
'''
def apply_masks(sess):
    ops = []
    for mask_tuple in masks:
        mask = mask_tuple[1]

        #mask_tuple[0] is the variable name,
        #name_tfv[var_name] gives the corresponding tf.var
        variable = name_tfv[mask_tuple[0]]
        new_var = tf.multiply(variable, mask)
        ops.append(tf.assign(variable, new_var))
    sess.run(ops)
'''

def apply_masks(sess):
    for mask_tuple in masks:
        mask = mask_tuple[1]
        variable = name_tfv[mask_tuple[0]]
        new_var = variable * mask
        sess.run(variable.assign(name_ph[mask_tuple[0]]), {name_ph[mask_tuple[0]]: sess.run(new_var)})
