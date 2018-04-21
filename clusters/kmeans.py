import numpy as np

from metric.metric import euclidean


class Kmeans:

    def __init__(self):
        pass

    @staticmethod
    def max_centroids(data_set, k):
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
                    dist_1 = euclidean(data_set[j, :], centroids[first, :])
                    dist_2 = euclidean(data_set[j, :], centroids[second, :])
                    dist = min(dist_1, dist_2)
                    if dist > max_dist:
                        max_dist = dist
                        index = j
            centroids[i, :] = data_set[index, :]
            visited.add(index)
            first = second
            second = second + 1

        return centroids

    @staticmethod
    def fit(data_set, k, dist_metric=euclidean, init_k_centroids=max_centroids):
        m, n = data_set.shape
        centroids = init_k_centroids(data_set=data_set, k=k)
        cluster_changed = True
        # cluster_assign is to store closest centroids index and distance
        cluster_assign = -np.ones((m, 2))
        while cluster_changed:
            cluster_changed = False
            for i in range(m):
                min_dist = np.inf
                min_index = cluster_assign[i, 0]
                for j in range(k):
                    distance = dist_metric(data_set[i, :], centroids[j, :])
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
                    centroids[j, :] = np.mean(belong_to_cluster, axis=0)

        print(euclidean(centroids[0, :], centroids[1, :]))
        return centroids, cluster_assign
