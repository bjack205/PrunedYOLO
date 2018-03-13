#This is the library for weights sparsification

import tensorflow as tf
import numpy as np
import os
import psutil
process = psutil.Process(os.getpid())

# masks is a list of tuple, each tuple is (var_name, mask)
# masks = []

# name_tfv is a dictionary, where key is the variable name,
# and value is tf.variable
# name_tfv = {}

# name_ph, key variable name and value is tf.placeholder (with same dtype and shape)
# name_ph = {}

def print_mem():
    print(str(process.memory_info().rss / 1024**2) + "kb")

def sparsify_network(sess, percent):
    # get_mask generates masks sparsifying model weights
    # Criteria for sparsification is specified by percientile,
    # i.e. fraction of elements masked
    # arguments:
    # sess: tf sess where weights are stored
    # percent: fraction of sparsified
    # print("Get variables")
    tf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # name_vals is list, each is tuple (var_name, value)
    name_vals = []
    data = []
    # print_mem()

    # print("Calculate variables")
    # old_vals only contains values, it's a list
    # old_vals = sess.run(tf_variables)
    # print_mem()

    # print("Concat into big array")
    # for val in old_vals:
    #     if len(val.shape) > 1:
    #         data.extend(np.absolute(val).flatten().tolist())

    calc_percentile = False
    prune_list = []
    for var in tf_variables:
        if len(var.shape) > 1:
            prune_list.append(var)
            if calc_percentile:
                data.extend(np.abs(sess.run(var).flatten()))

    # print_mem()

    if calc_percentile:
        # print("Calc cutoff")
        CUTOFF = np.percentile(data, percent)
        # print_mem()
        del data
    else:
        CUTOFF = 1e-6

    # print("Create placeholders")
    # for i in range(len(tf_variables)):
    #     if len(tf_variables[i].shape) > 1:
    #         name_tfv[tf_variables[i].name] = tf_variables[i]
    #         cur_ph = tf.placeholder(var.dtype, shape=var.get_shape())
    #         name_ph[var.name] = cur_ph
    #         name_vals.append((tf_variables[i].name, old_vals[i]))
    # print_mem()

    # print("Create masks")
    # global masks
    num_pruned = {}
    old_vals = sess.run(prune_list)
    for i, var in enumerate(prune_list):
        weight_val = old_vals[i]
        mask_cur = np.abs(weight_val) < CUTOFF
        weight_val *= mask_cur
        num_pruned[var.name] = np.sum(mask_cur)
        ph = tf.placeholder(var.dtype, shape=var.get_shape())
        """
        mask_cur = np.ones(weight_val.shape)
        shape_len = len(weight_val.shape)
        if shape_len > 1:
            for i in range(weight_val.shape[0]):
                for j in range(weight_val.shape[1]):
                    if shape_len > 2:
                        for k in range(weight_val.shape[2]):
                            if shape_len > 3:
                                for l in range(weight_val.shape[3]):
                                    if abs(weight_val[i, j, k, l]) < CUTOFF:
                                        mask_cur[i, j] = 0
                            else:
                                if abs(weight_val[i, j, k]) < CUTOFF:
                                    mask_cur[i, j] = 0
                    else:
                        if abs(weight_val[i, j]) < CUTOFF:
                            mask_cur[i, j] = 0
        """

        #pair[0] is the variable name

        #masks.append((var.name, mask_cur))
        sess.run(var.assign(ph), {ph: weight_val})
    print_mem()
    return num_pruned

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


def apply_masks(sess):
    print("Apply masks")
    for mask_tuple in masks:
        mask = mask_tuple[1]
        variable = name_tfv[mask_tuple[0]]
        new_var = variable * mask
        sess.run(variable.assign(name_ph[mask_tuple[0]]), {name_ph[mask_tuple[0]]: sess.run(new_var)})
'''