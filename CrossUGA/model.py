from layers import *
import tensorflow as tf
from initializations import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders,num_features,dis,**kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs1 = placeholders[0]['features']
        self.input_dim1 = num_features[0]
        self.features_nonzero1 = placeholders[0]['features_nonzero']
        self.adj1 = placeholders[0]['adj']

        self.inputs2 = placeholders[1]['features']
        self.input_dim2 = num_features[1]
        self.features_nonzero2 = placeholders[1]['features_nonzero']
        self.adj2 = placeholders[1]['adj']

        self.dropout = placeholders[0]['dropout']
        self.D_W1 = placeholders[0]['D_W1']
        self.D_b1 = placeholders[0]['D_b1']
        self.D_W2 = placeholders[0]['D_W2']
        self.D_b2 = placeholders[0]['D_b2']      
        self.build()

    def _build(self):
        self.hidden1,self.hidden2_ = GraphConvolutionSparse(input_dim=[self.input_dim1,self.input_dim2],
                                              output_dim=FLAGS.g_hidden1,
                                              adj=[self.adj1,self.adj2],
                                              features_nonzero=[self.features_nonzero1,self.features_nonzero2],
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)([self.inputs1,self.inputs2])


        self.embeddings1,self.embeddings2_ = GraphConvolution(input_dim=FLAGS.g_hidden1,
                                           output_dim=FLAGS.g_hidden2,
                                           adj=[self.adj1,self.adj2],
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)([self.hidden1,self.hidden2_])
        self.hidden2,self.hidden1_ = GraphConvolutionSparse(input_dim=[self.input_dim2,self.input_dim1],
                                              output_dim=FLAGS.g_hidden1,
                                              adj=[self.adj2,self.adj1],
                                              features_nonzero=[self.features_nonzero1,self.features_nonzero2],
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)([self.inputs2,self.inputs1])


        self.embeddings2,self.embeddings1_ = GraphConvolution(input_dim=FLAGS.g_hidden1,
                                           output_dim=FLAGS.g_hidden2,
                                           adj=[self.adj2,self.adj1],
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)([self.hidden2,self.hidden1_])

        self.fake_logits = dis_layer(D_w1=self.D_W1,D_b1=self.D_b1,D_w2=self.D_W2,D_b2=self.D_b2, 
                                  act=tf.nn.relu,
                                  logging=self.logging)(self.embeddings2_)
                                  
        self.fake_prob = tf.reduce_mean(self.fake_logits)

        self.attribute_decoder_layer1_1 = FullyConnectedDecoder(input_dim=FLAGS.g_hidden2,
                                           output_dim=self.input_dim1,
                                           adj=self.adj1,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.embeddings1)


        self.attribute_reconstructions1 = self.attribute_decoder_layer1_1


        self.reconstructions1 = InnerProductDecoder(input_dim=FLAGS.g_hidden2,
                                      act=lambda x: x,
                                      # act=tf.nn.sigmoid,
                                      logging=self.logging)(self.embeddings1)

        self.reconstructions2 = InnerProductDecoder(input_dim=FLAGS.g_hidden2,
                                      act=lambda x: x,
                                      # act=tf.nn.sigmoid,
                                      logging=self.logging)(self.embeddings2_)




    def distance_matrix(self,x1, x2, distance_metric):
        """Computes the pairwise distance between two tensors.
        Args:
            x1: A tensor of size [m, k].
            x2: A tensor of size [n, k].
            distance_metric: Either 'euclidean' or 'cosine'.
        Returns:
            A tensor of size [m, n] whose elements are the distances between x1 and
            x2.
        """
        if distance_metric == 'euclidean':
            l = tf.reduce_sum(x1**2, axis=1, keepdims=True)
            m = 2.0 * tf.tensordot(x1, x2, axes=[[1], [1]])
            r = tf.transpose(tf.reduce_sum(x2**2, axis=1, keepdims=True))
            return l - m + r
        elif distance_metric == 'cosine':
            x1 = tf.nn.l2_normalize(x1, axis=1)
            x2 = tf.nn.l2_normalize(x2, axis=1)
            # return 1.0 - tf.tensordot(x1, x2, axes=[[1], [1]])
            return tf.transpose(0.5 + 0.5 * tf.tensordot(x1, x2, axes=[[1], [1]]))
        else:
            raise ValueError('Unknown distance_metric: %s' % distance_metric)

    def cosine_similarity_matrix(self,a, b):
        """ 
        compute cosine similarity between v1 and v2 in matrix
        """
        normalize_a = tf.nn.l2_normalize(a, axis=1)
        normalize_b = tf.nn.l2_normalize(b, axis=1)
        return tf.matmul(normalize_a, tf.transpose(normalize_b))
class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)



class indentify_edge(object):
    def __init__(self, emb):
        self.emb = emb
        self.h_dim = 128
        self.num_nodes = self.emb.shape[0]
        self.num_feas = self.emb.shape[1]

        with tf.variable_scope('indentify_edge'):
            self.W1 = tf.Variable(self.xavier_init([self.num_feas,self.h_dim]))
            self.b1 = tf.Variable(tf.zeros([self.num_feas]))
            self.W2 = tf.Variable(self.xavier_init([self.h_dim,self.num_feas]))
            self.b2 = tf.Variable(tf.zeros([self.h_dim]))
        self.logits = tf.add(tf.matmul(tf.add(tf.matmul(self.emb,self,W1),self.b1),self.W2),self.b2)
        self.pre_label = tf.sigmoid(self.logits)





    def xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 1.)
        # xavier_stddev = 0.02
        return tf.random_normal(shape=size, stddev=xavier_stddev)


# def discriminator(real_emb,fake_emb):
#     real_h1 = tf.nn.leaky_relu(tf.matmul(real_emb,D2_W1) + D2_b1,0.01)
#     real_logits = tf.matmul(real_h1, D2_W2) + D2_b2
#     real_logits = tf.nn.sigmoid(real_logits)
#     fake_h1 = tf.nn.leaky_relu(tf.matmul(fake_emb,D2_W1) + D2_b1,0.01)
#     fake_logits = tf.matmul(fake_h1, D2_W2) + D2_b2
#     fake_logits = tf.nn.sigmoid(fake_logits)
#     return real_logits,fake_logits