#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import os
import string
import zipfile
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import collections
import itertools
import scipy
import heapq
from sklearn.preprocessing import robust_scale

num_of_features = 13


# the following function performs data preprocessing
def preprocessing(raw_text):
    proc_data = {}
    # List of stopwords
    stopwords = ['http', 'ειναι', 'ποιοι', 'ποιος', 'ποιους', 'αλλων', 'αλλους', 'ειμαστε', 'ακομα', 'αδιακοπα',
                 'ακομη',
                 'ακριβως', 'αλλα', 'αλλιως', 'αλλοτε', 'αμεσως', 'απεναντι', 'αντιπερα', 'αργοτερα', 'αριστερα']

    for key, value in raw_text.items():
        # Remove punctuation and lowercase
        punctuation = set(string.punctuation)
        doc = ''.join([w for w in value.lower() if w not in punctuation])

        # Νumber removal
        doc = [w for w in doc.split() if w.isalpha() and len(w) > 4]

        # removing emphasis
        doc = [w.replace('ά', 'α') for w in doc]
        doc = [w.replace('ό', 'ο') for w in doc]
        doc = [w.replace('έ', 'ε') for w in doc]
        doc = [w.replace('ή', 'η') for w in doc]
        doc = [w.replace('ί', 'ι') for w in doc]
        doc = [w.replace('ύ', 'υ') for w in doc]
        doc = [w.replace('ώ', 'ω') for w in doc]
        doc = [w.replace('ϊ', 'ι') for w in doc]
        doc = [w.replace('ϋ', 'υ') for w in doc]

        # Stopword removal
        doc = [w for w in doc if w not in stopwords]

        # Covenrt list of words to one string
        doc = ' '.join(w for w in doc)
        proc_data[key] = doc
    return proc_data


# open zip files and creates a dictionary with key in each host and value the union of the txt files
filenames = os.listdir('dataset/hosts')
raw_text = {}
# a list of host names
hosts_list = []

for zipfilename in filenames:
    with zipfile.ZipFile('dataset/hosts/' + zipfilename) as z:
        text = ""
        hosts_list.append(zipfilename[:-4])
        for filename in z.namelist():
            if not os.path.isdir(filename):
                with z.open(filename) as f:
                    for line in codecs.iterdecode(f, 'utf8'):
                        text += line
                        text += " "

        raw_text[zipfilename[:-4]] = text

print('hosts_list:', hosts_list)

proc_data = preprocessing(raw_text)


# for key, value in proc_data.items():
#     print(key, ':', value)
#     print()


# TF-IDF matrix construction of the dataset
# v = TfidfVectorizer()
# X = v.fit_transform(proc_data.values()).toarray()
# print(v.get_feature_names())
# print('Shape of X(tf-idf):', X.shape)
# print('Sparse:', np.count_nonzero(X) / float(X.shape[0] * X.shape[1]))


# Compute cosine similarity, S: array (numOfDocs X numOfDocs) of pairwise similarities between all documents
def compute_similarities(X):
    S = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            par = (np.linalg.norm(X[i, :]) * np.linalg.norm(X[j, :]))
            if par != 0:
                S[i, j] = np.dot(X[i, :], X[j, :]) / par
                S[j, i] = S[i, j]
            else:
                S[i, j] = 0
    return S


# cos_sim_matrix = compute_similarities(X)
# print('Cosine similarity:', cos_sim_matrix.shape)

# smatrix = open('pickles/cos_sim_matrix.pkl','wb')
# pickle.dump(cos_sim_matrix, smatrix)
# smatrix.close()
matrixs = open('pickles/cos_sim_matrix.pkl', 'rb')
S = pickle.load(matrixs)
matrixs.close()
print('S matrix stored')

dim = S.shape[0]
X_feat = np.zeros((S.shape[0] * S.shape[1], num_of_features))

# store data of edgelist.txt into a list and a dictionary where key=tuple and value=1 for connection
with open('dataset/edgelist.txt') as txt_file:
    con_list = []
    dict = {}
    for line in txt_file:
        v1, v2 = line.rstrip().split('\t')
        raw_list = []
        raw_list.append(v1)
        raw_list.append(v2)
        con_list.append(raw_list)
        dict[tuple(raw_list)] = 1
        raw_list.append(1)


# create a matrix to represent graph connections between all docs
def graph_connections(hosts_list, dim, dict):
    Z = np.zeros((dim, dim))
    for i in range(len(hosts_list)):
        for j in range(0, len(hosts_list)):
            tuple_list = []
            tuple_list.append(hosts_list[i])
            tuple_list.append(hosts_list[j])
            key_tuple = tuple(tuple_list)
            if key_tuple in dict:
                Z[i, j] = 1
    return Z


# Z = graph_connections(hosts_list, dim, dict)
# print()
# print('Z size:', Z.shape)
# print(Z)
# zmatrix = open('pickles/con_matrix.pkl','wb')
# pickle.dump(Z, zmatrix)
# zmatrix.close()
matrixz = open('pickles/con_matrix.pkl', 'rb')
Z = pickle.load(matrixz)
matrixz.close()


# create 2 dictionaries one with key(all possible tuple of hosts): value(row position) and opposite


def create_dict(hosts_list):
    dict_1 = {}
    dict_2 = {}
    row_position = 0
    for i in range(len(hosts_list)):
        for j in range(0, len(hosts_list)):
            tuple_list = []
            position = []
            position.append(i)
            position.append(j)
            position.append(row_position)
            tuple_position = tuple(position)
            tuple_position = row_position
            tuple_list.append(hosts_list[i])
            tuple_list.append(hosts_list[j])
            key_tuple = tuple(tuple_list)
            # if i != j:
            dict_1[key_tuple] = tuple_position
            dict_2[tuple_position] = key_tuple
            row_position += 1
    # print('dict_1:', dict_1)
    #print('dict_2:', dict_2)
    return dict_1


dict_1 = create_dict(hosts_list)

# pos = open('pickles/dict_2.pkl','wb')
# pickle.dump(dict_2, pos)
# pos.close()
pos = open('pickles/dict_2.pkl', 'rb')
dict_2 = pickle.load(pos)
pos.close()


# load dataset to a directed graph
G = nx.read_edgelist('dataset/edgelist.txt', delimiter='\t', create_using=nx.DiGraph())
nodes_list = nx.nodes(G)
print('**Nodes list**', nodes_list)
# print(len(nodes_list))


# *************************************************************************
# compute in_degree and in_degree number

def in_out_degree(G, dim):
    num_rows = 0
    out_degree_matrix = np.zeros((dim, 1))
    in_degree_matrix = np.zeros((dim, 1))
    for k in hosts_list:
        if k in nodes_list:
            out_degree = G.out_degree(k)
            in_degree = G.in_degree(k)
            out_degree_matrix[num_rows] = out_degree
            in_degree_matrix[num_rows] = in_degree
            num_rows += 1
    outdmatrix = open('pickles/out_degree_matrix.pkl', 'wb')
    pickle.dump(out_degree_matrix, outdmatrix)
    outdmatrix.close()
    indmatrix = open('pickles/in_degree_matrix.pkl', 'wb')
    pickle.dump(in_degree_matrix, indmatrix)
    indmatrix.close()


# in_out_degree(G, dim)
outdmatrix = open('pickles/out_degree_matrix.pkl', 'rb')
out_degree_matrix = pickle.load(outdmatrix)
outdmatrix.close()
indmatrix = open('pickles/in_degree_matrix.pkl', 'rb')
in_degree_matrix = pickle.load(indmatrix)
indmatrix.close()


# *************************************************************************
# compute core number


def core_calc(G, dim):
    core_table = np.zeros((dim, 1))
    core_num_dict = nx.core_number(G)
    z = 0
    for i in hosts_list:
        core_table[z] = core_num_dict.get(i)
        z += 1
    # print(core_table)
    return core_table


# core_table = core_calc(G, dim)
# print('Core Done')
#
# cmatrix = open('pickles/core_matrix.pkl','wb')
# pickle.dump(core_table, cmatrix)
# cmatrix.close()
cmatrix = open('pickles/core_matrix.pkl', 'rb')
core_table = pickle.load(cmatrix)
cmatrix.close()


# ****************************************************************************
# compute degree centrality


def degree_centrality(G, dim):
    degree_cent_matrix = np.zeros((dim, 1))
    degree_cent_dict = nx.degree_centrality(G)
    z = 0
    for i in hosts_list:
        degree_cent_matrix[z] = degree_cent_dict.get(i)
        z += 1
    return degree_cent_matrix


# degree_cent_matrix = degree_centrality(G, dim)
# print('degree centrality Done')
#
# dmatrix = open('pickles/degree_cent_matrix.pkl','wb')
# pickle.dump(degree_cent_matrix, dmatrix)
# dmatrix.close()
dmatrix = open('pickles/degree_cent_matrix.pkl', 'rb')
degree_cent_matrix = pickle.load(dmatrix)
dmatrix.close()


# ****************************************************************************
# compute pageRank


def pagerank(G, dim):
    pagerank_matrix = np.zeros((dim, 1))
    pagerank_dict = nx.pagerank(G)
    z = 0
    for i in hosts_list:
        pagerank_matrix[z] = pagerank_dict.get(i)
        z += 1
    return pagerank_matrix


# pagerank_matrix = pagerank(G, dim)
# print('Pagerank Done')

# pmatrix = open('pickles/pagerank_matrix.pkl','wb')
# pickle.dump(pagerank_matrix, pmatrix)
# pmatrix.close()
pmatrix = open('pickles/pagerank_matrix.pkl', 'rb')
pagerank_matrix = pickle.load(pmatrix)
pmatrix.close()


# ******************************************************************************
# compute in and out common neighbors for each tuple


def common_neighbors(hosts_list, dim, G):
    counter = 0
    common_in_neighbors = np.zeros((dim * dim, 1))
    common_out_neighbors = np.zeros((dim * dim, 1))
    for i in hosts_list:
        for j in hosts_list:
            if i == j:
                common_in = float(0)
                common_out = float(0)
                common_in_neighbors[counter] = common_in
                common_out_neighbors[counter] = common_out
            else:
                common_in = len(set(G.predecessors(i)).intersection(G.predecessors(j)))
                common_out = len(set(G.successors(i)).intersection(G.successors(j)))
                common_in_neighbors[counter] = common_in
                common_out_neighbors[counter] = common_out
            counter += 1
    inmatrix = open('pickles/common_in_neighbors_matrix.pkl', 'wb')
    pickle.dump(common_in_neighbors, inmatrix)
    inmatrix.close()
    outmatrix = open('pickles/common_out_neighbors_matrix.pkl', 'wb')
    pickle.dump(common_out_neighbors, outmatrix)
    outmatrix.close()


# common_neighbors(hosts_list, dim, G)
# print('Common neighbors Done')

inmatrix = open('pickles/common_in_neighbors_matrix.pkl', 'rb')
common_neigh_in_matrix = pickle.load(inmatrix)
inmatrix.close()
outmatrix = open('pickles/common_out_neighbors_matrix.pkl', 'rb')
common_neigh_out_matrix = pickle.load(outmatrix)
outmatrix.close()


# ************************************************************************
# compute betweenness centrality: the sum of the fraction of all-pairs shortest paths that pass through a node


def betweenness_centrality(G, dim):
    betweenness_cent_matrix = np.zeros((dim, 1))
    betweenness_cent_dict = nx.betweenness_centrality(G)
    z = 0
    for i in hosts_list:
        betweenness_cent_matrix[z] = betweenness_cent_dict.get(i)
        z += 1
    return betweenness_cent_matrix


# betweenness_cent_matrix = betweenness_centrality(G, dim)
# print('Betweenness centrality Done')

# bmatrix = open('pickles/betweenness_cent_matrix.pkl','wb')
# pickle.dump(betweenness_cent_matrix, bmatrix)
# bmatrix.close()
bmatrix = open('pickles/betweenness_cent_matrix.pkl', 'rb')
betweenness_cent_matrix = pickle.load(bmatrix)
bmatrix.close()


# *************************************************************************
# compute katz centrality


def katz_centrality(G, dim):
    katz_cent_matrix = np.zeros((dim, 1))
    katz_cent_dict = nx.katz_centrality(G)
    z = 0
    for i in hosts_list:
        katz_cent_matrix[z] = katz_cent_dict.get(i)
        z += 1
    return katz_cent_matrix


# katz_centrality_matrix = katz_centrality(G, dim)
# print('katz centrality Done')

# kmatrix = open('pickles/katz_cent_matrix.pkl','wb')
# pickle.dump(katz_centrality_matrix, kmatrix)
# kmatrix.close()
kmatrix = open('pickles/katz_cent_matrix.pkl', 'rb')
katz_cent_matrix = pickle.load(kmatrix)
kmatrix.close()


# *************************************************************************
# compute closeness centrality


def closeness_centrality(G, dim):
    closeness_cent_matrix = np.zeros((dim, 1))
    closeness_cent_dict = nx.closeness_centrality(G)
    z = 0
    for i in hosts_list:
        closeness_cent_matrix[z] = closeness_cent_dict.get(i)
        z += 1
    return closeness_cent_matrix


# closeness_cent_matrix = closeness_centrality(G, dim)
# print('closeness centrality Done')
# clmatrix = open('pickles/closeness_cent_matrix.pkl','wb')
# pickle.dump(closeness_cent_matrix, clmatrix)
# clmatrix.close()
clmatrix = open('pickles/closeness_cent_matrix.pkl', 'rb')
closeness_cent_matrix = pickle.load(clmatrix)
clmatrix.close()


# *************************************************************************
# compute shortest path length


def shortest_path(G, dim):
    shortest_path_matrix = np.zeros((dim*dim, 1))
    shortest_path_dict = nx.shortest_path_length(G)
    print(shortest_path_dict)
    for k, v in shortest_path_dict.items():
        l = []
        node_2_dict = shortest_path_dict.get(k)
        l.append(k)
        # print(node_2_dict)
        for key, value in node_2_dict.items():
            l.append(key)
            if tuple(l) in dict_1:
                pos = dict_1.get(tuple(l))
                # print(pos, tuple(l))
                shortest_path_matrix[pos] = node_2_dict.get(key)
            l.remove(key)
    # print(shortest_path_matrix)
    return shortest_path_matrix


# shortest_path_matrix = shortest_path(G, dim)
# print('Shortest path Done')

# spmatrix = open('pickles/shortest_path_matrix.pkl','wb')
# pickle.dump(shortest_path_matrix, spmatrix)
# spmatrix.close()
spmatrix = open('pickles/shortest_path_matrix.pkl', 'rb')
shortest_path_matrix = pickle.load(spmatrix)
spmatrix.close()


# *************************************************************************
# compute hits

def hits(G, dim):
    hubs_matrix = np.zeros((dim, 1))
    auth_matrix = np.zeros((dim, 1))
    hubs_dict, auth_dict = nx.hits(G)
    z = 0
    for i in hosts_list:
        hubs_matrix[z] = hubs_dict.get(i)
        auth_matrix[z] = auth_dict.get(i)
        z += 1
    hmatrix = open('pickles/hubs_matrix.pkl', 'wb')
    pickle.dump(hubs_matrix, hmatrix)
    hmatrix.close()
    amatrix = open('pickles/auth_matrix.pkl', 'wb')
    pickle.dump(auth_matrix, amatrix)
    amatrix.close()


# hits(G, dim)
# print('Hits')

hmatrix = open('pickles/hubs_matrix.pkl', 'rb')
hubs_matrix = pickle.load(hmatrix)
hmatrix.close()
amatrix = open('pickles/auth_matrix.pkl', 'rb')
auth_matrix = pickle.load(amatrix)
amatrix.close()


# *************************************************************************
# compute adamic adar index


def adamic_adar(G, dim):
    G2 = G.to_undirected()
    preds = nx.adamic_adar_index(G2)
    adamic_adar_matrix = np.zeros((dim*dim, 1))
    for u, v, p in preds:
        l = []
        l.append(u)
        l.append(v)
        if tuple(l) in dict_1:
            pos = dict_1.get(tuple(l))
            adamic_adar_matrix[pos] = p
    return adamic_adar_matrix

# adamic_adar_matrix = adamic_adar(G, dim)
# print('Adamic adar')

# aamatrix = open('pickles/adamic_adar_matrix.pkl','wb')
# pickle.dump(adamic_adar_matrix, aamatrix)
# aamatrix.close()
aamatrix = open('pickles/adamic_adar_matrix.pkl', 'rb')
adamic_adar_matrix = pickle.load(aamatrix)
aamatrix.close()


# *****************************************************************************
# compute Jaccard Similarity


def Jaccard_sim(proc_data, hosts_list, X_feat):
    counter = 0
    for i in hosts_list:
        for j in hosts_list:
            if i == j:
                jaccardcoefficient = float(0)
            else:
                a = frozenset(proc_data.get(i))
                b = frozenset(proc_data.get(j))
                jaccardcoefficient = (float(len(a & b)) / float(len(a | b)))
            X_feat[counter, 7] = jaccardcoefficient
            counter += 1
    return X_feat


# X_feat = Jaccard_sim(proc_data, hosts_list, X_feat)
# print('Jaccard done')

# jmatrix = open('pickles/jaccard_matrix.pkl','wb')
# pickle.dump(X_feat, jmatrix)
# jmatrix.close()
jmatrix = open('pickles/jaccard_matrix.pkl', 'rb')
jaccard_table = pickle.load(jmatrix)
jmatrix.close()

print('Features loaded....................')

# fill in the X_feat matrix with other features
count = 0
for j in range(out_degree_matrix.shape[0]):
    for i in range(out_degree_matrix.shape[0]):
        X_feat[count, 0] = S[j, i]
        X_feat[count, 1] = out_degree_matrix[j]
        X_feat[count, 2] = in_degree_matrix[j]
        X_feat[count, 3] = out_degree_matrix[i]
        X_feat[count, 4] = in_degree_matrix[i]
        X_feat[count, 5] = core_table[j]
        X_feat[count, 6] = core_table[i]
        # X_feat[count, 7] = jaccard_table[count, 7]
        X_feat[count, 7] = common_neigh_in_matrix[count]
        X_feat[count, 8] = common_neigh_out_matrix[count]
        # X_feat[count, 9] = betweenness_cent_matrix[j]
        # X_feat[count, 10] = betweenness_cent_matrix[i]
        X_feat[count, 9] = pagerank_matrix[j]
        X_feat[count, 10] = pagerank_matrix[i]
        X_feat[count, 11] = degree_cent_matrix[j]
        X_feat[count, 12] = degree_cent_matrix[i]
        # X_feat[count, 15] = katz_cent_matrix[j]
        # X_feat[count, 16] = katz_cent_matrix[i]
        # X_feat[count, 15] = closeness_cent_matrix[j]
        # X_feat[count, 16] = closeness_cent_matrix[i]
        # X_feat[count, 15] = shortest_path_matrix[count]
        # X_feat[count, 9] = adamic_adar_matrix[count]
        # X_feat[count, 13] = hubs_matrix[j]
        # X_feat[count, 14] = hubs_matrix[i]
        # X_feat[count, 10] = auth_matrix[j]
        # X_feat[count, 11] = auth_matrix[i]
        count += 1

print('X_feat table:', X_feat.shape)


# model learning *************************************************

Y = Z.flatten()
Y = np.reshape(Y, (Y.shape[0], 1))


def model_training(Y, X_feat):
    clf = LogisticRegression()
    clf.fit(X_feat, Y.ravel())
    # prediction = clf.predict(X_feat)
    # y_score is a matrix where col1:prob of 2 nodes not connected
    y_score = clf.predict_proba(X_feat)
    print()
    print('predict proba table:')
    print(y_score)
    return y_score

def forests_training(Z, X_feat):
    Y = Z.flatten()
    Y = np.reshape(Y, (Y.shape[0], 1))
    clf = RandomForestClassifier(n_estimators = 100, max_features= "log2",oob_score= True, random_state= 50,  max_leaf_nodes=25, min_samples_leaf = 50)
    clf.fit(X_feat, Y.ravel())
    y_score = clf.predict_proba(X_feat)
    print()
    print('predict proba table:')
    print(y_score)
    return y_score


y_score = forests_training(Z, X_feat)

# store probabilities
prob_dict = {}
prob_dictR = {}
for i in range(y_score.shape[0]):
    prob_dict[y_score[i, 1]] = i
    prob_dictR[i] = y_score[i, 1]


# create a new dict to store keys with the same values
new_dict = {}
for k, v in prob_dictR.items():
    new_dict.setdefault(v, []).append(k)
print('new dict', new_dict)

# sort the dict by key in descending order
od = collections.OrderedDict(sorted(new_dict.items(), reverse = True))
print('Ordered Dictionary:', od)

# list (of lists) with indexes of probs
lista = []
for k,v in od.items():
    lista.append(v)

# join the indexes into one list
index_list = list(itertools.chain.from_iterable(lista))


# create a list of indexes with 453 max values except existed ones
max_index_list = []
for i in index_list:
    if Y[i] == 0:
        max_index_list.append(i)
        if len(max_index_list) == 453:
            break

print('List of indexes with max probs:', max_index_list)
print(len(max_index_list))


num_missing_edges = 453
nodes = G.nodes()


# write predicted pairs of hosts in a txt file
with codecs.open('predicted_edges.txt', 'w', encoding='utf-8') as f:
    for i in max_index_list:
        node1 = dict_2.get(i)[0]
        node2 = dict_2.get(i)[1]
        f.write(node1 + '\t' + node2 + '\n')
