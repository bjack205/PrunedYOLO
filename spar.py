#This is the library for weights sparsification

import tensorflow as tf
import numpy as np

#masks is a list of tuple, each tuple is (var_name, mask)
masks = []

#name_tfv is a dictionary, where key is the variable name,
#and value is tf.variable
name_tfv = {}
feed_dict  = {}
ops = []

def get_masks(sess, percent):
    #get_mask generates masks sparsifying model weights
    #Criteria for sparsification is specified by percientile,
    #i.e. fraction of elements maksed
    #arguments:
    #sess: tf sess where weights are stored
    #percent: fraction of sparsified
    tf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    #name_vals is list, each is tuple (var_name, np value)
    name_vals = []
    data=[]

    # count pruning percentage by layer
    prune_percent = []

    #old_vals only contains values, it's a list
    old_vals = sess.run(tf_variables)
    for val in old_vals:
        if len(val.shape) > 1:
            data.extend(np.absolute(val).flatten().tolist())

    CUTOFF = np.percentile(data, percent)
    del data
    global feed_dict
    global ops

    for i in range(len(tf_variables)):
        if(len(old_vals[i].shape) > 1):
            name_tfv[tf_variables[i].name] = tf_variables[i]
            cur_ph = tf.placeholder(tf_variables[i].dtype, shape=tf_variables[i].get_shape())
            cur_op = tf_variables[i].assign(tf_variables[i]*cur_ph)
            name_vals.append((tf_variables[i].name, old_vals[i], cur_ph, cur_op))
            ops.append(cur_op)


    for pair in name_vals:
        weight_val = pair[1]
        mask_cur = weight_val >= CUTOFF
        prune_percent.append(np.sum(~mask_cur) / np.size(mask_cur))
        '''
        mask_cur = np.ones(weight_val.shape)
        if len(weight_val.shape)==2:
            for i in range(weight_val.shape[0]):
                for j in range(weight_val.shape[1]):
                    if abs(weight_val[i, j]) <  CUTOFF:
                        mask_cur[i ,j] = 0
        '''

        feed_dict[pair[2]] = mask_cur

    return prune_percent

def apply_masks(sess):
    sess.run(ops, feed_dict)
