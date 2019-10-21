from __future__ import division
from __future__ import print_function
import random
import numpy as np

np.random.seed(123)

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    adj_list -- list: id to nbs
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, adj_list, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25, val_ratio = 0.5,
            **kwargs):

        self.adj_list = adj_list.tocoo()
        self.node_size = adj_list.shape[0]
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(np.arange(self.node_size))#

        self.all_edges, _, _ = self.split_edges()

        self.adj, self.deg = self.construct_adj()
        print('minibach done')
        # self.test_adj = self.construct_test_adj() unimplemented.


    def split_edges(self, val_ratio=0.6):
        edges = []
        train_edges = []
        val_edges = []

        for index in range(self.adj_list.nnz):
            edge = (self.adj_list.row[index],self.adj_list.col[index])
            edges.append(edge)
            if random.random() < val_ratio:
                val_edges.append(edge)
            else:
                train_edges.append(edge)
        return edges, train_edges, val_edges

    def construct_adj(self):
        adj = self.node_size*np.ones((self.node_size+1, self.max_degree))
        deg = np.zeros((self.node_size,))

        neighbors = []
        for nodeid in range(self.node_size):
            neighbors.append([])

        for edge in self.all_edges:
            neighbors[edge[0]].append(edge[1])

        for nodeid in range(self.node_size):
            nbs = neighbors[nodeid]
            deg[nodeid] = len(nbs)
            if len(nbs) == 0:
                continue
            if len(nbs) > self.max_degree:
                nbs = np.random.choice(nbs, self.max_degree, replace=False)
            elif len(nbs) < self.max_degree:
                nbs = np.random.choice(nbs, self.max_degree, replace=True)
            adj[nodeid, :] = nbs
        return adj, deg


    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(node1)
            batch2.append(node2)

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = np.arange(self.node_size)
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges


    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.all_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    adj_list -- networkx adj_list
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, adj_list, 
            placeholders, label_map, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):

        self.adj_list = adj_list
        self.nodes = adj_list.nodes()
        self.node_size = len(self.nodes)        
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.adj_list.nodes() if self.adj_list.node[n]['val']]
        self.test_nodes = [n for n in self.adj_list.nodes() if self.adj_list.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(adj_list.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[n] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = self.node_size*np.ones((self.node_size+1, self.max_degree))
        deg = np.zeros((self.node_size,))

        for nodeid in self.adj_list.nodes():
            if self.adj_list.node[nodeid]['test'] or self.adj_list.node[nodeid]['val']:
                continue
            neighbors = np.array([neighbor for neighbor in self.adj_list.neighbors(nodeid)
                if (not self.adj_list[nodeid][neighbor]['train_removed'])])
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = self.node_size*np.ones((self.node_size+1, self.max_degree))
        for nodeid in self.adj_list.nodes():
            neighbors = np.array([neighbor 
                for neighbor in self.adj_list.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [n for n in batch1id]
              
        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
