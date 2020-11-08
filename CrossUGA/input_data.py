import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.io import loadmat
import tensorflow as tf
#from sklearn.preprocessing import normalize

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# adj: <class 'scipy.sparse.csr.csr_matrix'>
# feature: <class 'scipy.sparse.lil.lil_matrix'>
def load_data(dataset):
    if 'Douban' in dataset:
        data = loadmat('./data/Douban/' + dataset + '.mat')
        adj = sp.csr_matrix(data['A'])
        fea = sp.lil_matrix(data['X'])
        # print(type(fea))
        fea = preprocess_features(fea)
        # print(fea)        
        # print(type(fea))
        # print(fea.shape)

    # elif 'flickr' in dataset:
    #     data = loadmat('./data/flickr/' + dataset + '.mat')
    #     adj = sp.csr_matrix(data['A'])
    #     fea = sp.lil_matrix(data['X'])
    elif 'flickr' in dataset:
        data = loadmat('./data/flickr_lzp/' + dataset + '.mat')
        # data = loadmat('./data/' + dataset + '/' + dataset + '.mat')        
        adj = sp.csr_matrix(data['A'])
        fea = sp.lil_matrix(data['X'])    
    elif 'lastfm' in dataset:
        data = loadmat('./data/lastfm_myspace/' + dataset + '.mat')
        # data = loadmat('./data/' + dataset + '/' + dataset + '.mat')
        adj = sp.csr_matrix(data['A'])
        fea = sp.lil_matrix(data['X'])
        fea = sp.lil_matrix(preprocess_features(fea))           
    elif 'myspace' in dataset:
        data = loadmat('./data/lastfm_myspace/' + dataset + '.mat')
        # data = loadmat('./data/' + dataset + '/' + dataset + '.mat')
        adj = sp.csr_matrix(data['A'])
        fea = sp.lil_matrix(data['X']) 
        fea = sp.lil_matrix(preprocess_features(fea))
    elif 'wiki' in dataset:
        data = loadmat('./data/wiki/' + dataset + '.mat')
        # data = loadmat('./data/' + dataset + '/' + dataset + '.mat')
        adj = sp.csr_matrix(data['A'])
        fea = sp.lil_matrix(data['X']) 
    elif 'blogs' in dataset: #blogs
        data = np.loadtxt('./data/blogs/' + dataset + '.txt',dtype=np.int32)
        row = [] 
        col = []
        value = []
        for ele in data:
            row.append(ele[0]-1)
            col.append(ele[1]-1)
            value.append(1)
        adj = sp.csr_matrix((value,(row,col)),shape=(max(row)+1,max(row)+1))
        fea = sp.identity(max(row)+1)  # featureless
        # print(fea.shape)

    else:
        data = np.loadtxt('./data/dblp/' + dataset + '.txt',dtype=np.int32)
        row = [] 
        col = []
        value = []
        for ele in data:
            row.append(ele[0]-1)
            col.append(ele[1]-1)
            value.append(1)
        adj = sp.csr_matrix((value,(row,col)),shape=(max(col)+1,max(col)+1))
        fea = sp.identity(max(col)+1)
    return adj,fea,fea.shape[1]


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten() #into one-hot
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx