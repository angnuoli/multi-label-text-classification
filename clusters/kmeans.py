import numpy as np

class Kmeans:

    def __init__(self):
        pass

    def euclidean(self, x_, y_):
        return np.linalg.norm(x_ - y_)

    def max_centroids(self, dataSet, k):
        m, n = dataSet.shape
        centroids = np.zeros((k, n))
        first_index = np.random.randint(k)
        centroids[0, :] = dataSet[first_index, :]
        visited = set()
        visited.add(first_index)

        first = second = 0
        for i in range(1, k):
            max_dist = 0
            index = -1
            for j in range(m):
                if j not in visited:
                    dist_1 = self.euclidean(dataSet[j, :], centroids[first, :])
                    dist_2 = self.euclidean(dataSet[j, :], centroids[second, :])
                    dist = min(dist_1, dist_2)
                    if dist > max_dist:
                        max_dist = dist
                        index = j
            centroids[i, :] = dataSet[index, :]
            visited.add(index)
            first = second
            second = second + 1

        return centroids

    def fit(self, dataSet, k, dist_metric=euclidean, init_k_centroids= max_centroids):
        m, n = dataSet.shape
        centroids = init_k_centroids(self, dataSet=dataSet, k=k)
        cluster_changed = True
        # cluster_assign is to store closest centroids index and distance
        cluster_assign = -np.ones((m, 2))
        while cluster_changed:
            cluster_changed = False
            for i in range(m):
                min_dist = np.inf
                min_index = cluster_assign[i, 0]
                for j in range(k):
                    distance = dist_metric(self, dataSet[i, :], centroids[j, :])
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
                    belong_to_cluster = dataSet[cluster_assign[:, 0] == j]
                    centroids[j, :] = np.mean(belong_to_cluster, axis=0)

        print(self.euclidean(centroids[0, :], centroids[1, :]))
        return centroids, cluster_assign