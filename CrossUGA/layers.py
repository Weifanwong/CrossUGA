from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj1 = adj[0]
        self.adj2 = adj[1]
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj1, x)
        # x = tf.layers.batch_normalization(x, training=False)        
        outputs1 = self.act(x)

        y = inputs[1]
        y = tf.nn.dropout(y, 1-self.dropout)
        y = tf.matmul(y, self.vars['weights'])
        y = tf.sparse_tensor_dense_matmul(self.adj2, y)
        # y = tf.layers.batch_normalization(y, training=False)        
        outputs2 = self.act(y)        
        return [outputs1,outputs2]


class GraphConvolution2(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution2, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj1 = adj[0]
        self.adj2 = adj[1]
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj1, x)
        # x = tf.layers.batch_normalization(x, training=False)        
        outputs1 = tf.nn.leaky_relu(x,0.01)

        y = inputs[1]
        y = tf.nn.dropout(y, 1-self.dropout)
        y = tf.matmul(y, self.vars['weights'])
        y = tf.sparse_tensor_dense_matmul(self.adj2, y)
        # y = tf.layers.batch_normalization(y, training=False)        
        outputs2 = tf.nn.leaky_relu(y,0.01)
        return [outputs1,outputs2]


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim[0], output_dim, name="weights")
                                
        self.dropout = dropout
        self.adj1 = adj[0]
        self.adj2 = adj[1]
        self.act = act
        self.issparse = True
        self.features_nonzero1 = features_nonzero[0]
        self.features_nonzero2 = features_nonzero[1]

    def _call(self, inputs):
        x = inputs[0]
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj1, x)
        # x = tf.layers.batch_normalization(x, training=False)
        #outputs1 = self.act(x)
        outputs1 = tf.nn.leaky_relu(x,0.01)
        y = inputs[1]
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        y = tf.sparse_tensor_dense_matmul(y, self.vars['weights'])
        y = tf.sparse_tensor_dense_matmul(self.adj2, y)
        # y = tf.layers.batch_normalization(y, training=False)

        outputs2 =  tf.nn.leaky_relu(y,0.01) 
        return [outputs1,outputs2]

class FullyConnectedDecoder(Layer):
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(FullyConnectedDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        outputs = tf.matmul(x, self.vars['weights'])
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class InnerProductDecoder2(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder2, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs1 = tf.nn.dropout(inputs[0], 1-self.dropout)
        inputs2 = tf.nn.dropout(inputs[1], 1-self.dropout)
        # inputs1 = tf.nn.l2_normalize(inputs1)
        # inputs2 = tf.nn.l2_normalize(inputs2)
        x = tf.transpose(inputs[0])
        x = tf.matmul(inputs[1], x)
        # x = tf.reshape(x, [-1])
        outputs = self.act(x)
        #outputs = x
        return outputs




class dis_layer(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self,D_w1,D_b1,D_w2,D_b2, act=tf.nn.sigmoid, **kwargs):
        super(dis_layer, self).__init__(**kwargs)
        self.act = act
        self.D_W1 = D_w1
        self.D_W2 = D_w2
        self.D_b1 = D_b1
        self.D_b2 = D_b2

    def _call(self, inputs):
        real_h1 = tf.nn.leaky_relu(tf.matmul(inputs,self.D_W1) + self.D_b1,0.01)
        real_logits = tf.matmul(real_h1, self.D_W2) + self.D_b2
        # real_logits = tf.nn.sigmoid(real_logits)
        # outputs = tf.nn.sigmoid(real_logits)
        return real_logits
