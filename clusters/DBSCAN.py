from random import choice

import numpy as np

from data_structure.data_structure import StaticData
from metric.metric import euclidean


class DBSCAN:
    """This is DBSCAN cluster"""

    def __init__(self, epsilon=5, min_pts=5):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.edges = {}
        self.core_points = set()
        self.border_points = set()
        self.noise_points = set()

    @staticmethod
    def find_epsilon(X):
        k_distance = []
        m = X.shape[0]

        for i in range(m):
            for j in range(i + 1, m):
                k_distance.append(euclidean(X[i], X[j]))

        k_distance = sorted(k_distance)
        StaticData.dbscan_k_distance = k_distance

    def preprocess(self, X):
        """
        This method is to initialize the distance matrix between each point.
        The matrix is formatted as dict(dict()).

        :param X:
        :return:
        """

        m = X.shape[0]

        print("Compute the distance matrix between each point...")
        if len(StaticData.edges) == 0:
            edges = {}
            for i in range(m):
                edges[i] = {}

            for i in range(m):
                for j in range(i + 1, m):
                    edges[j][i] = edges[i][j] = np.linalg.norm(X[i] - X[j])

            StaticData.edges = edges

        edges = {}

        for i in range(m):
            edges[i] = {}

        for i in range(m):
            for j in range(i+1, m):
                if StaticData.edges[i][j] <= self.epsilon:
                    edges[j][i] = edges[i][j] = StaticData.edges[i][j]

            if len(edges[i].keys()) >= self.min_pts:
                self.core_points.add(i)

        self.edges = edges
        return edges

    @staticmethod
    def purity_in_class(index, labels):
        freq = {}

        for i in index:
            for label in labels[i]:
                if label not in freq:
                    freq[label] = 0
                freq[label] += 1

        p = 0.0
        label_ = ''
        for label in freq.keys():
            if p < freq[label]:
                p = freq[label]
                label_ = label

        StaticData.dbscan_labels.append(label_)
        return p / len(index)

    def calculate_purity(self, clusters, y):
        purity = 0.0
        total = 0.0

        for i, classes in clusters.items():
            m = float(len(classes))
            total += m

            purity += m * self.purity_in_class(clusters[i], y)

        return purity / total

    def fit(self, X):
        matrix = self.preprocess(X)
        core_points = set(self.core_points)
        k = 0
        m = X.shape[0]
        unvisited = set()
        for i in range(m):
            unvisited.add(i)
        clusters = {}

        while len(core_points) != 0:
            k_core_point = choice(list(core_points))
            cluster_k = set()
            cur_core_points_queue = set()

            cur_core_points_queue.add(k_core_point)
            core_points.remove(k_core_point)
            cluster_k.add(k_core_point)
            unvisited.remove(k_core_point)

            while len(cur_core_points_queue) != 0:
                core_point = choice(list(cur_core_points_queue))
                cur_core_points_queue.remove(core_point)
                neighbors = unvisited & matrix[core_point].keys()
                cluster_k = cluster_k | neighbors
                unvisited = unvisited - neighbors
                cur_core_points_queue = cur_core_points_queue | (neighbors & core_points)

            clusters[k] = cluster_k
            k += 1
            core_points = core_points - cluster_k

        return clusters
