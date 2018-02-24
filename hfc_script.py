'''
This is the script for hierarchical fully connected layer function script
'''
import tensorflow as tf
import numpy as np
from collections import defaultdict
from utils import get_ind

def meshpart(tree, n_lvls):
    #if n_lvls > 9: n_lvls = n_lvls - 3
    for lvl_cur in range(1, n_lvls):
        for ptr in tree[lvl_cur]:
            if len(ptr) < 2:continue
            tree[lvl_cur+1].append(ptr[:int(len(ptr)/2)])
            tree[lvl_cur+1].append(ptr[int(len(ptr)/2):])
    return tree

def hfc(in_nodes, in_size, out_size, bt_size, rank = 2):
    '''
    The input nodes is the layer before fc one,
    w/ size [batch, input_size].
    out_size is the fully connected layer size.
    '''
    xi = tf.contrib.layers.xavier_initializer()
    in_tree = defaultdict(lambda: [])
    out_tree = defaultdict(lambda: [])
    in_tree[1] = [range(in_size)]
    out_tree[1] = [range(out_size)]
    batch_size = tf.shape(in_nodes)[0]

    #n_lvls is the height of tree
    n_lvls = int(np.ceil(np.log2(min(in_size, out_size))))
    if n_lvls > 9: n_lvls = n_lvls - 3
    #The in_tree and out_tree are two dicts rep. for binary tree
    out_tree = meshpart(out_tree, n_lvls)
    in_tree = meshpart(in_tree, n_lvls)
    #return out_nodes as final result
    '''
    zeros_dims = tf.stack([in_size, tf.shape(in_nodes)[0]])
    out_nodes = tf.Variable(tf.fill(zeros_dims, 0.0), trainable = False)
    '''
    out_nodes = tf.zeros([bt_size, out_size], tf.float32)
    in_nodes_t = tf.transpose(in_nodes)
    for lvl_cur in range(1, n_lvls + 1):
        length = min(len(in_tree[lvl_cur]), len(out_tree[lvl_cur]))
        for i in range(length):
            dim_in = len(in_tree[lvl_cur][i])
            dim_out = len(out_tree[lvl_cur][i])
            if min(dim_in, dim_out) < 2:rank = 1
            wb = tf.get_variable("wb_{0}{1}".format(lvl_cur, i), [dim_in, rank], initializer=xi)
            bb = tf.get_variable('bb_{0}{1}'.format(lvl_cur, i), [rank, rank], initializer=xi)
            bw = tf.get_variable("bw_{0}{1}".format(lvl_cur, i), [rank, dim_out], initializer=xi)

            temp1 = tf.matmul(tf.transpose(tf.gather(in_nodes_t, np.asarray(in_tree[lvl_cur][i]))), wb)
            temp2 = tf.matmul(temp1, bb)
            updates = tf.matmul(temp2, bw)
            '''
            ext = np.zeros([dim_out, out_size], np.float32)
            ext[range(dim_out), out_tree[lvl_cur][i]] = 1
            '''
            ext = tf.SparseTensor(indices = get_ind(out_tree[lvl_cur][i]), values = [1.0]*dim_out, dense_shape = [dim_out, out_size])
            out_nodes += tf.transpose(tf.sparse_tensor_dense_matmul(b=updates, sp_a = ext, adjoint_a=True, adjoint_b=True))
            #tf.scatter_add(ref = out_nodes, indices = out_tree[lvl_cur][i], updates = tf.transpose(updates))
            #out_nodes[:, out_tree[lvl_cur][i]] += tf.matmul(temp2, bw)
            rank = 2

    #imitate the sparse matrices in the last layer
    for i in range(length):
        dim_in = len(in_tree[lvl_cur][i])
        dim_out = len(out_tree[lvl_cur][i])
        ww = tf.get_variable("ww_{0}{1}".format(lvl_cur, i), [dim_in, dim_out], initializer=xi)
        '''
        ext = np.zeros([dim_out, out_size], np.float32)
        ext[range(dim_out), out_tree[lvl_cur][i]] = 1
        '''

        updates = tf.matmul(tf.transpose(tf.gather(in_nodes_t, np.asarray(in_tree[lvl_cur][i]))), ww)
        ext = tf.SparseTensor(indices = get_ind(out_tree[lvl_cur][i]), values = [1.0]*dim_out, dense_shape = [dim_out, out_size])
        out_nodes += tf.transpose(tf.sparse_tensor_dense_matmul(b=updates, sp_a = ext, adjoint_a=True, adjoint_b=True))
        #updates = tf.matmul(tf.transpose(tf.gather(in_nodes_t, in_tree[lvl_cur][i])), ww)
        #tf.scatter_add(ref = out_nodes, indices = out_tree[lvl_cur][i], updates = tf.transpose(updates))

    return out_nodes
