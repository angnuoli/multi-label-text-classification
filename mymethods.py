import csv
import os
import time

import numpy as np

from classifier.knn_classifier import KNNClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier
from clusters.DBSCAN import DBSCAN
from clusters.kmeans import Kmeans
from data_structure.data_structure import StaticData
from metric.metric import calculate_tf_idf, add_value, calculate_purity


def derivative_feature_vectors(_vocabulary):
    """Set up the fixed length feature vector.

    :param _vocabulary:
    :return: feature vector with artificial cardinality
    """

    feature_vector_1_tmp = {}
    feature_vector_2_tmp = {}
    for feature in _vocabulary[0:min(len(_vocabulary), 125)]:
        feature_vector_1_tmp[feature] = len(feature_vector_1_tmp)
    for feature in _vocabulary[0:min(len(_vocabulary), 270)]:
        feature_vector_2_tmp[feature] = len(feature_vector_2_tmp)
    return feature_vector_1_tmp, feature_vector_2_tmp


def generate_tf_idf_feature(bag_of_features, raw_documents):
    m = len(bag_of_features)
    feature_matrix = np.zeros((len(raw_documents), len(bag_of_features)))
    i = 0
    for _document in raw_documents:
        feature_vector = np.zeros(m)
        for feature, col in bag_of_features.items():
            if feature in _document.tfs['all']:
                tf = _document.tfs['all'][feature]
                df = StaticData.df_term[feature]
                tf_idf = calculate_tf_idf(tf=tf, df=df, doc_num=StaticData.n_train_documents)
                add_value(_document.tf_idf, feature, tf_idf)
                feature_vector[col] = tf_idf

        _document.feature_vector = feature_vector
        feature_matrix[i] = feature_vector
        i += 1
    return feature_matrix


def calculate_accuracy(y_predict, y_test_origin):
    """ Calculate the accuracy of Y_predict. The cardinality of Y_predict and Y_test_origin is same.

    accuracy = avg∑(true labels in predicted class labels) / len(predicted class labels)

    :param y_predict: [[class labels]] a series of class label of articles
    :param y_test_origin: [[test labels]] original class labels
    :return: accuracy
    """
    _accuracy = 0.0
    for y1, y2 in zip(y_predict, y_test_origin):
        counter = 0
        for y in y1:
            if y in y2:
                counter += 1
        _accuracy += float(counter) / len(y1)
    _accuracy /= len(y_predict)
    return _accuracy


def knn_predict(feature_vector,
                train_documents,
                test_documents,
                feature_matrix,
                y_test_original):
    knn_classifier = KNNClassifier(df_of_classes=StaticData.df_of_classes)
    StaticData.knn_build_time.append(time.time() - StaticData.A1)
    print("Offline efficiency cost - time to build knn model: {} s."
          .format(StaticData.knn_build_time[len(StaticData.knn_build_time) - 1]))

    y_knn_predict, y_knn_accuracy = [], 0.0
    for k in range(5, 6):
        knn_classifier.k = k
        StaticData.A1 = time.time()
        y_knn_predict = knn_classifier.knn_predict(feature_vector,
                                                   train_documents,
                                                   test_documents,
                                                   feature_matrix)
        StaticData.knn_predict_time.append(time.time() - StaticData.A1)
        print("Online efficiency cost - time to predict: {} s."
              .format(StaticData.knn_predict_time[len(StaticData.knn_predict_time) - 1]))
        y_knn_accuracy = calculate_accuracy(y_knn_predict,
                                            y_test_original)
        StaticData.knn_accuracy.append(y_knn_accuracy)
        print("\nKNN Classifier: The number of neighbors : k is {}, accuracy is: {}.".format(k, y_knn_accuracy))
    return y_knn_predict, y_knn_accuracy


def naive_predict(feature_vector,
                  vocabulary_,
                  train_documents,
                  test_documents,
                  y_test_original):
    StaticData.A1 = time.time()
    naive_classifier = NaiveBayesClassifier(feature_vector=feature_vector,
                                            vocabulary=vocabulary_,
                                            n=len(train_documents))
    naive_classifier.fit(train_documents)
    StaticData.naive_build_time.append(time.time() - StaticData.A1)
    print("Offline efficiency cost - time to build naive model: {} s."
          .format(StaticData.naive_build_time[len(StaticData.naive_build_time) - 1]))

    StaticData.A1 = time.time()
    y_predict = naive_classifier.predict(test_documents, k=0)
    StaticData.naive_predict_time.append(time.time() - StaticData.A1)
    print("Online efficiency cost - time to predict using naive model: {} s."
          .format(StaticData.naive_predict_time[len(StaticData.naive_predict_time) - 1]))
    y_accuracy = calculate_accuracy(y_predict, y_test_original)

    StaticData.naiver_accuracy.append(y_accuracy)
    print("\nWhen the cardinality of feature vector is {}, "
          "the accuracy of Naive Bayes Classifier is {}."
          .format(len(feature_vector), y_accuracy))
    return y_predict


def kmeans_cluster(feature_vector,
                   y_train_original,
                   test_feature_matrix,
                   feature_matrix,
                   y_test_original):
    kmeas_cluster_ = Kmeans()
    labels = y_train_original
    purity = []

    for k in range(2, 44, 10):
        StaticData.A1 = time.time()
        centroids, cluster_assign = kmeas_cluster_.fit(dataSet=feature_matrix, k=k)
        purity.append(calculate_purity(centroids, cluster_assign, labels, k))

        StaticData.kmeans_cluster_time.append(time.time() - StaticData.A1)
        print("Time to cluster: {} s."
              .format(StaticData.kmeans_cluster_time[len(StaticData.kmeans_cluster_time) - 1]))

        print("\nKmeans Cluster: The number of clusters : k - k is {}, purity is: {}.".format(k, purity))
    StaticData.kmeans_purity = purity
    return purity


def dbscan_cluster(train_feature_matrix,
                   y_train_original,
                   test_feature_matrix=None,
                   y_test_original=None,
                   epsilon=5,
                   min_pts=5):
    print("epsilon: {}, min_pts: {}".format(epsilon, min_pts))
    dbscan_cluster = DBSCAN(epsilon, min_pts)
    # dbscan_cluster.find_epsilon(train_feature_matrix)
    clusters = dbscan_cluster.fit(train_feature_matrix, y_train_original)
    purity = dbscan_cluster.calculate_purity(clusters, y_train_original)
    return clusters, purity


def generate_dataset(documents, vocab):
    # check whether the subdirectory exists or not if not create a subdirectory
    subdirectory = "output"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    print("Start writing data to vocabulary.csv")
    with open('output/vocabulary.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Term", "Index"])
        writer.writerows(vocab.items())
    print("Finish writing data to vocabulary.csv")
    print("Start writing data to dataset.csv")
    with open('output/dataset.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["document_id - (feature, vector) - [class labels]"])
        writer.writerow('')
        document_id = 0
        for document in documents:
            print("Writing document {}".format(document_id))
            document.id = document_id
            writer.writerow(["document {}".format(document_id)])
            writer.writerow(["class labels:"])
            writer.writerow(document.class_list)
            writer.writerow(['feature vector:'])
            rows = []
            for feature, frequency in document.tfs['all'].items():
                output_str = "({},{})".format(feature, frequency)
                rows.append(output_str)
            writer.writerow(rows)
            writer.writerow('')
            document_id += 1
    print("Finish writing data to dataset.csv")


def write_to_file(iterator, filename):
    # check whether the subdirectory exists or not if not create a subdirectory
    subdirectory = "output"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    print("\nStart writing data to {}...".format(filename))
    with open('output/' + filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows([iterator])
        csv_file.close()
    print("Finish writing data to {}.\n".format(filename))


def write_predict(y_original, y_predict, filename):
    # check whether the subdirectory exists or not if not create a subdirectory
    subdirectory = "output"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    print("\nStart writing data to {}...".format(filename))
    with open('output/' + filename, 'w', newline='') as writer:
        writer.writelines("True labels -> Predicted labels\n")
        for Y1, Y2 in zip(y_original, y_predict):
            writer.writelines("{} -> {}\n".format(Y1, Y2))
        writer.close()
    print("Finish writing data to {}.\n".format(filename))


def write_termination_messages(filename):
    # check whether the subdirectory exists or not if not create a subdirectory
    subdirectory = "output"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    print("\nStart writing data to {}...".format(filename))
    with open('output/' + filename, 'w', newline='') as writer:
        writer.writelines("========== Termination message ==========\n")
        writer.writelines("Mission completed.\n")
        writer.writelines("We select knn classifier and naive classifier.\n")
        writer.writelines("\nFor feature vector with {} cardinality:\n".format(125))
        writer.writelines("\nThe accuracy of knn classifier is {}.\n".format(StaticData.knn_accuracy[0]))
        writer.writelines("The offline efficient cost of knn classifier is {} s.\n"
                          .format(StaticData.knn_build_time[0]))
        writer.writelines("The online efficient cost of knn classifier is {} s.\n"
                          .format(StaticData.knn_predict_time[0]))
        writer.writelines("\nThe accuracy of naive classifier is {}.\n".format(StaticData.naiver_accuracy[0]))
        writer.writelines(
            "The offline efficient cost of naive classifier is {} s.\n".format(StaticData.naive_build_time[0]))
        writer.writelines(
            "The online efficient cost of naive classifier is {} s.\n".format(StaticData.naive_predict_time[0]))
        writer.writelines("\nFor feature vector with {} cardinality:\n".format(270))
        writer.writelines("\nThe accuracy of knn classifier is {}.\n".format(StaticData.knn_accuracy[1]))
        writer.writelines("The offline efficient cost of knn classifier is {} s.\n"
                          .format(StaticData.knn_build_time[1]))
        writer.writelines("The online efficient cost of knn classifier is {} s.\n"
                          .format(StaticData.knn_predict_time[1]))
        writer.writelines("\nThe accuracy of naive classifier is {}.\n".format(StaticData.naiver_accuracy[1]))
        writer.writelines("The offline efficient cost of naive classifier is {} s.\n"
                          .format(StaticData.naive_build_time[1]))
        writer.writelines("The online efficient cost of naive classifier is {} s.\n"
                          .format(StaticData.naive_predict_time[1]))
        writer.close()
    print("Finish writing data to {}.\n".format(filename))


# pipeline
def knn_predict_showcase(feature_vector_1, feature_vector_2, train_documents, test_documents, Y_test_original):
    """ knn predict """
    print("\n++++++++++ Start predicting ++++++++++")
    print("Select two classifiers: knn classifier and naive bayes classifier.")
    print("\n========== KNN Classifier ==========")
    print("Select k = 5 as the number of neighbors.")
    print("\nPredict using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    print("")
    StaticData.A1 = time.time()
    feature_matrix_1 = generate_tf_idf_feature(feature_vector_1, train_documents)

    Y_knn_128_predict, Y_knn_128_accuracy = knn_predict(feature_vector=feature_vector_1,
                                                        train_documents=train_documents,
                                                        test_documents=test_documents,
                                                        feature_matrix=feature_matrix_1,
                                                        y_test_original=Y_test_original)
    write_predict(Y_test_original, Y_knn_128_predict, "KNN_predict_class_labels_125_feature_vector.txt")
    print("\nPredict using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    StaticData.A1 = time.time()
    feature_matrix_2 = generate_tf_idf_feature(feature_vector_2, train_documents)
    Y_knn_256_predict, Y_knn_256_accuracy = knn_predict(feature_vector=feature_vector_2,
                                                        train_documents=train_documents,
                                                        test_documents=test_documents,
                                                        feature_matrix=feature_matrix_2,
                                                        y_test_original=Y_test_original)
    write_predict(Y_test_original, Y_knn_256_predict, "KNN_predict_class_labels_270_feature_vector.txt")


def naive_predict_showcase(feature_vector_1, feature_vector_2, vocabulary_, train_documents, test_documents,
                           Y_test_original):
    """ Naive Bayes predict """
    print("\n========== Naive Bayes Classifier ==========")
    print("\nPredict using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    Y_naive_128_predict = naive_predict(feature_vector_1,
                                        vocabulary_,
                                        train_documents,
                                        test_documents,
                                        Y_test_original)
    write_predict(Y_test_original, Y_naive_128_predict, "Naive_predict_class_labels_125_feature_vector.txt")
    print("\nPredict using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    Y_naive_256_predict = naive_predict(feature_vector_2,
                                        vocabulary_,
                                        train_documents,
                                        test_documents,
                                        Y_test_original)
    write_predict(Y_test_original, Y_naive_256_predict, "Naive_predict_class_labels_270_feature_vector.txt")

    print("\n========== Termination message ==========")
    print("Mission completed.")
    print("We select knn classifier and naive classifier.")
    print("\nFor feature vector with {} cardinality:".format(len(feature_vector_1)))
    print("\nThe accuracy of knn classifier is {}.".format(StaticData.knn_accuracy[0]))
    print("The offline efficient cost of knn classifier is {} s.".format(StaticData.knn_build_time[0]))
    print("The online efficient cost of knn classifier is {} s.".format(StaticData.knn_predict_time[0]))

    print("\nThe accuracy of naive classifier is {}.".format(StaticData.naiver_accuracy[0]))
    print("The offline efficient cost of naive classifier is {} s.".format(StaticData.naive_build_time[0]))
    print("The online efficient cost of naive classifier is {} s.".format(StaticData.naive_predict_time[0]))

    print("\nFor feature vector with {} cardinality:".format(len(feature_vector_2)))
    print("\nThe accuracy of knn classifier is {}.".format(StaticData.knn_accuracy[1]))
    print("The offline efficient cost of knn classifier is {} s.".format(StaticData.knn_build_time[1]))
    print("The online efficient cost of knn classifier is {} s.".format(StaticData.knn_predict_time[1]))

    print("\nThe accuracy of naive classifier is {}.".format(StaticData.naiver_accuracy[1]))
    print("The offline efficient cost of naive classifier is {} s.".format(StaticData.naive_build_time[1]))
    print("The online efficient cost of naive classifier is {} s.".format(StaticData.naive_predict_time[1]))

    write_termination_messages("termination_messages.txt")


def kmeans_showcase(feature_vector_1, feature_vector_2, Y_train_original, Y_test_original, train_documents,
                    test_documents):
    """Kmeans clustering"""
    print("\n++++++++++ Start clustering ++++++++++")
    print("Select two clusters: kmeans cluster.")
    print("\n========== Kmeans Cluster ==========")
    print("Select k = 5 as the number of neighbors.")
    print("\nCluster using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    print("")
    StaticData.A1 = time.time()
    feature_matrix_1 = generate_tf_idf_feature(feature_vector_1, train_documents)

    kmeans_cluster(feature_vector=feature_vector_1,
                   y_train_original=Y_train_original,
                   test_feature_matrix=feature_matrix_1,
                   feature_matrix=feature_matrix_1,
                   y_test_original=Y_test_original)

    print("\nCluster using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    StaticData.A1 = time.time()
    feature_matrix_2 = generate_tf_idf_feature(feature_vector_2, train_documents)
    kmeans_cluster(feature_vector=feature_vector_2,
                   y_train_original=Y_train_original,
                   test_feature_matrix=feature_matrix_2,
                   feature_matrix=feature_matrix_2,
                   y_test_original=Y_test_original)
