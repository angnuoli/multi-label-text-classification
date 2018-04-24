import os
import time
from sys import argv

from src.data_preprocess.preprocess import DataProcessor, MyVectorizer
from src.data_structure.data_structure import StaticData
from src.metric.metric import calculate_purity
from src.mymethods import derivative_feature_vectors, generate_tf_idf_feature, write_to_file, \
    dbscan_cluster, kmeans_cluster, assign_to_centroids, knn_predict_showcase, naive_predict_showcase

if __name__ == "__main__":

    A1 = time.time()
    if len(argv) > 1:
        data_dir = argv[1]
    else:
        data_dir = 'data'

    if not os.path.exists(data_dir):
        raise OSError('Please store original data files in data/ directory or type '
                      '"python3 main.py data_path" to input path of data')

    # =============================================================================
    #   Preprocessing
    # =============================================================================
    print("========== Parse data files ==========")
    data_dir = os.path.abspath(data_dir)
    data_processor = DataProcessor()
    train_documents, test_documents = data_processor.data_preprocess(data_dir)

    # test_documents = test_documents[0:10]
    # binarize the class label to class vectors
    print("\n========== Constructing bag of words ==========")
    count_vectorizer = MyVectorizer(max_df=0.9)
    train_documents, vocabulary_ = count_vectorizer.fit_transform(train_documents)
    count_vectorizer_test = MyVectorizer(max_df=0.9)
    count_vectorizer_test.count_vocab(test_documents)

    # get the Y test
    Y_test_original = []
    Y_train_original = []
    for document in test_documents:
        Y_test_original.append(document.class_['topics'])
    # get the Y train
    for document in train_documents:
        Y_train_original.append(document.class_['topics'])

    StaticData.preprocessing_time = time.time() - A1
    print("Preprocessing time: {} s.".format(StaticData.preprocessing_time))

    # construct more than 256 cardinality and less than 128 cardinality feature vectors
    print("\n========== Generate two derivative feature vectors of selected feature vector ==========")
    feature_vector_1, feature_vector_2 = derivative_feature_vectors(vocabulary_)
    print("Generate feature vector 1 which has a cardinality 125...")
    write_to_file(feature_vector_1, "feature_vector_1.csv")
    print("Generate feature vector 2 which has a cardinality 270...")
    write_to_file(feature_vector_2, "feature_vector_2.csv")

    # compute two feature matrix
    feature_matrix_1 = generate_tf_idf_feature(feature_vector_1, train_documents)
    feature_matrix_2 = generate_tf_idf_feature(feature_vector_2, train_documents)
    test_feature_matrix_1 = generate_tf_idf_feature(feature_vector_1, test_documents)
    test_feature_matrix_2 = generate_tf_idf_feature(feature_vector_2, test_documents)

    print("\n++++++++++ Start classifying ++++++++++")
    print("Select two classifiers: knn classifier and naive bayes classifier.")

    knn_predict_showcase(feature_vector_1, feature_vector_2, train_documents, test_documents, Y_test_original)

    naive_predict_showcase(feature_vector_1, feature_vector_2, vocabulary_, train_documents, test_documents,
                           Y_test_original)

    print("\n++++++++++ Start clustering ++++++++++")
    print("Select two clusters: kmeans cluster and DBSCAN cluster.")

    # =============================================================================
    #   Kmeans clustering
    # =============================================================================

    print("\n========== Kmeans Cluster ==========")
    k = 70
    print("Select k = {} as the number of clusters.".format(k))
    print("\nCluster using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    print("")
    StaticData.A1 = time.time()

    centroids = kmeans_cluster(y_train_original=Y_train_original,
                               feature_matrix=feature_matrix_1,
                               k=k
                               )
    clusters_assign = assign_to_centroids(centroids, test_feature_matrix_1)
    p1 = calculate_purity(cluster_assign_=clusters_assign, labels=Y_test_original, k=k)
    print("Apply the cluster to test data, the purity is: {}.".format(p1))

    print("\nCluster using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    StaticData.A1 = time.time()

    centroids = kmeans_cluster(y_train_original=Y_train_original,
                               feature_matrix=feature_matrix_2,
                               k=k
                               )
    clusters_assign = assign_to_centroids(centroids, test_feature_matrix_2)
    p1 = calculate_purity(cluster_assign_=clusters_assign, labels=Y_test_original, k=k)
    print("Apply the cluster to test data, the purity is: {}.".format(p1))

    # =============================================================================
    #   DBSCAN clustering
    # =============================================================================
    print("\n========== DBSCAN Cluster ==========")

    epsilon = 4
    min_pts = 1

    print("Select epsilon = {} and MinPts = {}.".format(epsilon, min_pts))
    print("\nCluster using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    print("")

    StaticData.A1 = time.time()
    clusters, p = dbscan_cluster(train_feature_matrix=feature_matrix_1, y_train_original=Y_train_original,
                                 epsilon=epsilon, min_pts=min_pts)
    print("Time to cluster: {} s."
          .format(StaticData.DBSCAN_cluster_time[len(StaticData.DBSCAN_cluster_time)-1]))
    print("The purity is {}.".format(p))
    print("clusters num: {}".format(len(clusters)))

    StaticData.edges = {}

    epsilon = 4
    min_pts = 1

    print("\nCluster using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    print("")

    StaticData.A1 = time.time()
    clusters, p = dbscan_cluster(train_feature_matrix=feature_matrix_2, y_train_original=Y_train_original,
                                 epsilon=epsilon, min_pts=min_pts)
    print("Time to cluster: {} s."
          .format(StaticData.DBSCAN_cluster_time[len(StaticData.DBSCAN_cluster_time) - 1]))
    print("The purity is {}.".format(p))
    print("clusters num: {}".format(len(clusters)))

    print("\n========== Termination message ==========")
    print("Mission completed.")
    print("We select kmeans and DBSCAN cluster.")
    print("\nFor feature vector with {} cardinality:".format(len(feature_vector_1)))
    print("\nThe purity of kmeans cluster is {}.".format(StaticData.kmeans_purity[0]))
    print("The efficient cost of kmeans cluster is {} s.".format(StaticData.kmeans_cluster_time[0]))

    print("\nThe purity of DBSCAN cluster is {}.".format(StaticData.DBSCAN_purity[0]))
    print("The efficient cost of DBSCAN cluster is {} s.".format(StaticData.DBSCAN_cluster_time[0]))

    print("\nFor feature vector with {} cardinality:".format(len(feature_vector_2)))
    print("\nThe purity of kmeans cluster is {}.".format(StaticData.kmeans_purity[1]))
    print("The efficient cost of kmeans cluster is {} s.".format(StaticData.kmeans_cluster_time[1]))

    print("\nThe purity of DBSCAN cluster is {}.".format(StaticData.DBSCAN_purity[1]))
    print("The efficient cost of DBSCAN cluster is {} s.".format(StaticData.DBSCAN_cluster_time[1]))
