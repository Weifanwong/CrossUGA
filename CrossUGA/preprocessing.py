import numpy as np
import scipy.sparse as sp
import random
import tensorflow as tf
import heapq

flags = tf.app.flags
FLAGS = flags.FLAGS

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features,pos_weight,norm,placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders[0]['features']: features[0]})
    feed_dict.update({placeholders[0]['adj']: adj_normalized[0]})
    feed_dict.update({placeholders[0]['adj_orig']: adj[0]})
    feed_dict.update({placeholders[0]['pos_weight']: pos_weight[0]})
    feed_dict.update({placeholders[0]['norm']: norm[0]})

    feed_dict.update({placeholders[1]['features']: features[1]})
    feed_dict.update({placeholders[1]['adj']: adj_normalized[1]})
    feed_dict.update({placeholders[1]['adj_orig']: adj[1]})
    feed_dict.update({placeholders[1]['pos_weight']: pos_weight[1]})
    feed_dict.update({placeholders[1]['norm']: norm[1]})

    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_data_for_classifer(adj,adj_dict,emb):
    adj_coo = adj.tocoo()
    col =list(adj_coo.col.reshape(-1))
    row =list(adj_coo.row.reshape(-1))
    adj_list = []
    for i in range(len(col)):
        adj_list.append([row[i],col[i]])
    random.shuffle(adj_list)
    real_batch = adj_list[0:FLAGS.batch_num2]
    fake_batch = []
    for i in range(len(real_batch)):
        flag = True
        while flag == True:
            tmp = random.randint(0,emb.shape[0]-1)
            if tmp not in adj_dict[real_batch[i][0]]:
                fake_batch.append([real_batch[i][0],tmp])
                flag = False
    real_batch_emb = []
    fake_batch_emb = []
    for edge in real_batch:
        first_node_emb = list(emb[edge[0]])
        second_node_emb = list(emb[edge[1]])
        tmp = first_node_emb + second_node_emb
        real_batch_emb.append(tmp)
    for edge in fake_batch:
        first_node_emb = list(emb[edge[0]])
        second_node_emb = list(emb[edge[1]])
        tmp = first_node_emb + second_node_emb
        fake_batch_emb.append(tmp)
    return real_batch_emb,fake_batch_emb

def get_data_for_gan(batch_num,node_1,node_2):
    rand_arr = np.arange(node_1)
    np.random.shuffle(rand_arr)
    emb_id_1 = [rand_arr[0:batch_num]]  
    rand_arr = np.arange(node_2)
    np.random.shuffle(rand_arr)
    s_batch = self.s_embeddings[rand_arr[0:batch_num]]
    return s_batch,t_batch


def adj2dict(adj):
    adj_dict = {}
    adj_coo = adj.tocoo()
    col =list(adj_coo.col.reshape(-1))
    row =list(adj_coo.row.reshape(-1))
    for i in range(0,adj.shape[0]):
        adj_dict[i] = []
    for i in range(len(row)):
        adj_dict[row[i]].append(col[i])
    return adj_dict


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 1.)
    # xavier_stddev = 0.02
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis


def cal_acc(final_emb1_1,final_emb2_1,final_emb1_2,final_emb2_2,topK):
    similar_matrix2_1 = EuclideanDistance(final_emb1_2,final_emb2_2)
    similar_matrix1_2 = EuclideanDistance(final_emb1_1,final_emb2_1)
    # print(similar_matrix1_2.shape)
    # print(similar_matrix2_1.shape)
    final_pair = []
    for index in range(similar_matrix1_2.shape[0]):
        cur_line1_2 = similar_matrix1_2[index]
        cur_line2_1 = similar_matrix2_1[index]
        cur_pair = []
        topkk = topK
        while len(cur_pair) < topK:
            top_index1_2 = heapq.nsmallest(topkk, range(len(cur_line1_2)), cur_line1_2.take)
            top_index1_2 = list(map(lambda x:x+1,top_index1_2))
            top_index2_1 = heapq.nsmallest(topkk, range(len(cur_line2_1)), cur_line2_1.take)
            top_index2_1 = list(map(lambda x:x+1,top_index2_1))
            cur_pair = list(set(top_index1_2).intersection(set(top_index2_1)))
            topkk += 1
        final_pair.append([index+1,cur_pair])
    return final_pair



def cosine_Matrix(_matrixA, _matrixB):
    _matrixA_matrixB = _matrixA * _matrixB.transpose()
    _matrixA_norm = numpy.sqrt(numpy.multiply(_matrixA,_matrixA).sum(axis=1))
    _matrixB_norm = numpy.sqrt(numpy.multiply(_matrixB,_matrixB).sum(axis=1))
    return numpy.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())

def cosine_similarity(matrix1,matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return 0.5 + 0.5 * cosine_distance

def innerproduct(matrix1,matrix2):
    x = np.dot(matrix2 , np.transpose(matrix1)) 
    return 1 / (1 + np.exp(-x))
    # return x



# def sigmoid(x):
#     return 

def get_local_AX(adj,feature,batch_num):
    node_num = adj.shape[0]
    id_list = list(range(node_num))
    np.random.shuffle(id_list)
    feature = feature.toarray()
    sample_A = []
    sample_X = []
    batches = int(node_num / batch_num)
    for i in range(batches):
        sample_A_cur = np.zeros((batch_num,batch_num))
        sample_X_cur = np.zeros((batch_num,feature.shape[1]))
        cur_id = id_list[i * batch_num : (i+1) * batch_num]
        for j in range(batch_num):
            for k in range(batch_num):
                sample_A_cur[j,k] = adj[cur_id[j],cur_id[k]]
            sample_X_cur[j] = feature[cur_id[j]]
        sample_A.append(sp.csr_matrix(sample_A_cur))
        sample_X.append(sp.lil_matrix(sample_X_cur))
    least_node_num = node_num - batches * batch_num
    cur_id = id_list[batches * batch_num:]
    for j in range(len(cur_id)):
        sample_A_cur = np.zeros((len(cur_id),len(cur_id)))
        sample_X_cur = np.zeros((len(cur_id),feature.shape[1]))
        for k in range(len(cur_id)):
            sample_A_cur[j,k] = adj[cur_id[j],cur_id[k]]
        sample_X_cur[j] = feature[cur_id[j]]
    sample_A.append(sp.csr_matrix(sample_A_cur))
    sample_X.append(sp.lil_matrix(sample_X_cur))

    return sample_A, sample_X, id_list



def softmax(adj):
    copy = adj
    adj_row_max = adj.max(axis=1)
    adj = adj - adj_row_max.reshape([copy.shape[0],1])
    adj_exp = np.exp(adj)
    adj_exp_row_sum = adj_exp.sum(axis=1).reshape([copy.shape[0],1])
    softmax = adj_exp / adj_exp_row_sum
    return softmax

def mean(adj):
    copy = adj
    adj_row_max = adj.max(axis=1)
    adj = adj - adj_row_max.reshape([copy.shape[0],1])
    adj_exp = np.exp(adj)
    adj_exp_row_sum = adj_exp.sum(axis=1).reshape([copy.shape[0],1])
    softmax = adj_exp / adj_exp_row_sum
    return softmax

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma    


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range    

def softmax_vec(x):
    totalSum = np.sum(np.exp(x), axis = 0)
    return np.exp(x)/totalSum    

def cal_mean_distance(similar_matrix):
    mean_distance = 0
    for i in range(similar_matrix.shape[0]):
        copy = similar_matrix[i].tolist()
        copy.sort(reverse=False)        
        print(copy[1:5])
        max_score = copy[0]  # 与top1与该点的相似度
        mean_distance += max_score
    mean_distance /= similar_matrix.shape[0]
    return mean_distance

def sample_network(adj,percent):
    print(adj.shape)
    remove_edges_list = []
    fake_edges_list = []
    rows = adj.nonzero()[0]
    cols = adj.nonzero()[1]
    edges_list = []    
    for i in range(len(rows)):
        edges_list.append([rows[i],cols[i]])
    edges_num = len(rows)
    number_of_remove_edges = int(percent * edges_num)
    remove_edges_index= random.sample(range(len(cols)),number_of_remove_edges)
    remove_edges_index.sort(reverse=True)
    rows_S = list(rows)
    cols_S = list(cols)
    for ele in remove_edges_index:
        # print(ele)
        remove_edges_list.append([rows_S[ele],cols_S[ele]])
        rows_S.pop(ele)
        cols_S.pop(ele)
    values = np.ones(len(rows_S))
    adj = sp.csr_matrix((values, (rows_S, cols_S)), shape=(adj.shape[0],adj.shape[0]))
    for i in range(number_of_remove_edges):
        fake_edges_pair = random.sample(range(adj.shape[0]),2)
        while fake_edges_pair in edges_list or [fake_edges_pair[1],s[0]] in edges_list:
            fake_edges_pair = random.sample(range(adj.shape[0]),2)
        fake_edges_list.append(fake_edges_pair)
    return adj,remove_edges_list,fake_edges_list
