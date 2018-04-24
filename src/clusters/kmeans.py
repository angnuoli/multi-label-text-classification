import time

import numpy as np


class Kmeans:

    def __init__(self):
        pass

    @staticmethod
    def max_centroids(data_set, k):
        print("Initialize the centroids for kmeans cluster...")
        A1 = time.time()
        m, n = data_set.shape
        centroids = np.zeros((k, n))
        first_index = np.random.randint(k)
        centroids[0, :] = data_set[first_index, :]
        visited = set()
        visited.add(first_index)

        first = second = 0
        for i in range(1, k):
            max_dist = 0
            index = -1
            for j in range(m):
                if j not in visited:
                    dist_1 = np.linalg.norm(data_set[j, :] - centroids[first, :])
                    dist_2 = np.linalg.norm(data_set[j, :] - centroids[second, :])
                    dist = min(dist_1, dist_2)
                    if dist > max_dist:
                        max_dist = dist
                        index = j
            centroids[i, :] = data_set[index, :]
            visited.add(index)
            first = second
            second = second + 1

        print("Finish initializing the centroids for kmeans cluster.")
        print("Cost time: {}s.".format(time.time() - A1))
        return centroids

    @staticmethod
    def random_centroids(data_set, k):
        print("Initialize the centroids for kmeans cluster...")
        A1 = time.time()
        m, n = data_set.shape
        centroids = np.zeros((k, n))

        index = list(range(len(data_set)))
        np.random.shuffle(index)

        for i in range(k):
            centroids[i, :] = data_set[index[i], :]

        print("Finish initializing the centroids for kmeans cluster.")
        print("Cost time: {}s.".format(time.time() - A1))
        return centroids

    def fit(self, data_set, k):
        m, n = data_set.shape
        centroids = self.random_centroids(data_set=data_set, k=k)
        cluster_changed = True
        # cluster_assign is to store closest centroids index and distance
        cluster_assign = -np.ones((m, 2))
        reassign_turn = 0
        while cluster_changed:
            reassign_turn += 1
            print("Moving centroids: {} turn.".format(reassign_turn))
            cluster_changed = False
            for i in range(m):
                min_dist = np.inf
                min_index = cluster_assign[i, 0]
                for j in range(k):
                    distance = np.linalg.norm(data_set[i, :] - centroids[j, :])
                    if distance < min_dist:
                        min_index = j
                        min_dist = distance
                if min_index != cluster_assign[i, 0]:
                    cluster_changed = True
                    cluster_assign[i, 0] = min_index
                    cluster_assign[i, 1] = min_dist

            # reassign centroids
            if cluster_changed:
                for j in range(k):
                    belong_to_cluster = data_set[cluster_assign[:, 0] == j]
                    if len(belong_to_cluster) != 0:
                        centroids[j, :] = np.mean(belong_to_cluster, axis=0)

        print("Finish kmeans clustering.")
        return centroids, cluster_assign
