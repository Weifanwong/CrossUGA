from __future__ import division
from __future__ import print_function

import time
import os
import heapq

# Train on CPU (hide GPU) due to memory constraints


import tensorflow as tf
import numpy as np
import scipy.sparse as sp

#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import *
from scipy.io import loadmat,mmwrite
from scipy import linalg
from scipy.io import loadmat,savemat
import networkx as nx
import heapq

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class main():
    def runner(self):
        model_str = FLAGS.model
        placeholders = [{
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features':tf.placeholder(tf.float32),
            'features_nonzero':tf.placeholder(tf.float32),
            'pos_weight':tf.placeholder(tf.float32),
            'norm':tf.placeholder(tf.float32),
            'reward':tf.placeholder(tf.float32),
            'D_W1':tf.placeholder_with_default(tf.zeros([FLAGS.g_hidden2,FLAGS.d_hidden1]),shape=[FLAGS.g_hidden2,FLAGS.d_hidden1]),
            'D_W2':tf.placeholder_with_default(tf.zeros([FLAGS.d_hidden1,1]),shape=[FLAGS.d_hidden1,1]),
            'D_b1':tf.placeholder_with_default(tf.zeros([FLAGS.d_hidden1]),shape=[FLAGS.d_hidden1]),
            'D_b2':tf.placeholder_with_default(tf.zeros([1]),shape=[1]),
        },
        {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features':tf.sparse_placeholder(tf.float32),
            'features_nonzero':tf.placeholder(tf.float32),
            'pos_weight':tf.placeholder(tf.float32),
            'norm':tf.placeholder(tf.float32),
            'reward':tf.placeholder(tf.float32)

        }]
        sess = tf.Session()


        real_X = tf.placeholder(tf.float32, shape=[None,FLAGS.g_hidden2])
        fake_X = tf.placeholder(tf.float32, shape=[None,FLAGS.g_hidden2])

        self.D_W1 = tf.Variable(xavier_init([FLAGS.g_hidden2,FLAGS.d_hidden1]))
        self.D_b1 = tf.Variable(xavier_init([FLAGS.d_hidden1]))
        self.D_W2 = tf.Variable(xavier_init([FLAGS.d_hidden1,1]))
        self.D_b2 = tf.Variable(xavier_init([1]))     
        d_vars = [self.D_W1,self.D_b1,self.D_W2,self.D_b2]


        print('train for the network embedding...')
        # Load data
        dataset_str1 = 'Douban_offline' # 1118 nodes
        dataset_str2 = 'Douban_online' # 3906 nodes
        adj1, features1,fea_num1 = load_data(dataset_str1)
        adj2, features2,fea_num2 = load_data(dataset_str2)
        num_features = [features1.shape[1],features2.shape[1]]

        model = None

        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders,num_features,sess)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

        # Optimizer

        with tf.name_scope('optimizer'):
            opt = OptimizerAE(preds=[model.reconstructions1,model.reconstructions2],
                              labels=[tf.reshape(tf.sparse_tensor_to_dense(placeholders[0]['adj_orig'],
                                                                                                    validate_indices=False), [-1]),
                              tf.reshape(tf.sparse_tensor_to_dense(placeholders[1]['adj_orig'],
                                                                                                    validate_indices=False), [-1])],
                              preds_attribute=[model.attribute_reconstructions1,model.attribute_reconstructions1],
                              labels_attribute=[tf.sparse_tensor_to_dense(placeholders[0]['features']),tf.sparse_tensor_to_dense(placeholders[1]['features'])],
                              pos_weight=[placeholders[0]['pos_weight'],placeholders[1]['pos_weight']],
                              norm=[placeholders[0]['norm'],placeholders[1]['norm']],
                              fake_logits = model.fake_logits,
                              alpha=FLAGS.AX_alpha)

        real_X = tf.placeholder(tf.float32, shape=[None,FLAGS.g_hidden2])
        fake_X = tf.placeholder(tf.float32, shape=[None,FLAGS.g_hidden2])

        real_logits,fake_logits = self.discriminator(real_X,fake_X)
        real_prob = tf.reduce_mean(real_logits)
        fake_prob = tf.reduce_mean(fake_logits)
        D_loss = - real_prob + fake_prob
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_dis)  # Adam Optimizer
        opt_dis = dis_optimizer.minimize(D_loss,var_list=d_vars)


        sess.run(tf.global_variables_initializer())
        final_emb1 = []
        final_emb2 = []
        emb1_id = []
        emb2_id = []
        local_A_1 = adj1
        local_X_1 = features1
        local_A_2 = adj2
        local_X_2 = features2

        adj_norm_1 = preprocess_graph(local_A_1)
        local_X_1 = sparse_to_tuple(local_X_1.tocoo())
        pos_weight_1 = float(local_A_1.shape[0] * local_A_1.shape[0] - local_A_1.sum()) / local_A_1.sum()
        adj_label_1 = local_A_1 + sp.eye(local_A_1.shape[0])
        adj_label_1 = sparse_to_tuple(adj_label_1)
        norm_1 = local_A_1.shape[0] * local_A_1.shape[0] / float((local_A_1.shape[0] * local_A_1.shape[0] - local_A_1.sum()) * 2)
        
        adj_norm_2 = preprocess_graph(local_A_2)
        local_X_2 = sparse_to_tuple(local_X_2.tocoo())
        pos_weight_2 = float(local_A_2.shape[0] * local_A_2.shape[0] - local_A_2.sum()) / local_A_2.sum()
        adj_label_2 = local_A_2 + sp.eye(local_A_2.shape[0])
        adj_label_2 = sparse_to_tuple(adj_label_2)
        norm_2 = local_A_2.shape[0] * local_A_2.shape[0] / float((local_A_2.shape[0] * local_A_2.shape[0] - local_A_2.sum()) * 2)
        
        self.tmp_count = {}

        for epoch in range(FLAGS.epoch):
          for circle_epoch in range(FLAGS.circle_epoch):
              for G_epoch in range(FLAGS.g_epoch):
              # ------------------------------------------------------------------------------------------
                feed_dict = construct_feed_dict([adj_norm_2,adj_norm_1], [adj_label_2,adj_label_1], [local_X_2,local_X_1], [pos_weight_2,pos_weight_1], [norm_2,norm_1],placeholders)          
                feed_dict.update({placeholders[0]['D_W1']: sess.run(self.D_W1)})
                feed_dict.update({placeholders[0]['D_W2']: sess.run(self.D_W2)})
                feed_dict.update({placeholders[0]['D_b1']: sess.run(self.D_b1)})
                feed_dict.update({placeholders[0]['D_b2']: sess.run(self.D_b2)})              

                _,embeddings1_,embeddings2_,gcn_cost,fake_prob_,attr_cost = sess.run([opt.opt_op,model.embeddings1,model.embeddings2_,opt.cost,model.fake_prob,opt.attribute_cost], feed_dict=feed_dict)

              for D_epoch in range(FLAGS.d_epoch):  
                feed_dict.update({placeholders[0]['dropout']: FLAGS.dropout})
                emb1,emb2 = sess.run([model.embeddings1,model.embeddings2_],feed_dict=feed_dict)
                _,real_prob_,fake_prob_ = sess.run([opt_dis,real_prob,fake_prob],feed_dict={real_X:emb1,fake_X:emb2})            

          if epoch % 1 == 0:

            emb1,emb2 = sess.run([model.embeddings1,model.embeddings2_],feed_dict=feed_dict)
            final_emb1 = np.array(emb1)
            final_emb2 = np.array(emb2)

            similar_matrix = cosine_similarity(final_emb1,final_emb2)

            self.similar_matrix = similar_matrix

            
            pair = {}
            gnd = np.loadtxt("data/douban_truth.emb")
            count = {}
            topk = [1,5,10,20,30,50]
            for i in range(len(topk)):
                pair[topk[i]] = []
                count[topk[i]] = 0
                self.tmp_count[topk[i]] = 0
            for top in topk:
                for index in range(similar_matrix.shape[0]):
                    top_index = heapq.nlargest(int(top), range(len(similar_matrix[index])), similar_matrix[index].take)        
                    top_index = list(map(lambda x:x+1,top_index))
                    pair[top].append([index+1,top_index])
                for ele_1 in gnd:
                  for ele_2 in pair[top]:
                    if ele_1[0] == ele_2[0]:
                      if ele_1[1] in ele_2[1]:
                        count[top] +=1


            print(f'-----------------------epoch {epoch}------------------------')
            for top in topk:
                print("top", '%02d' %(top), "count=", '%d' % (count[top]), "precision=", "{:.5f}".format(count[top]/len(gnd)))
            print(f'-----------------------epoch {epoch}------------------------')

    def discriminator(self,real_emb,fake_emb):
        real_h1 = tf.nn.leaky_relu(tf.matmul(real_emb,self.D_W1) + self.D_b1,0.01)
        real_logits = tf.matmul(real_h1, self.D_W2) + self.D_b2
        fake_h1 = tf.nn.leaky_relu(tf.matmul(fake_emb,self.D_W1) + self.D_b1,0.01)
        fake_logits = tf.matmul(fake_h1, self.D_W2) + self.D_b2
        return real_logits,fake_logits