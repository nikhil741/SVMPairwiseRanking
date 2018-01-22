##########Created By: Nikhil Agrawal, Shubham Yadav##########
###############Assignment_No.: 3, GroupNo.: 11###############

import re
from collections import Counter
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from math import log
import os

'''
    Function
        Calculates scores for pair consists of 5 axis and stores in a list and return list of all feautres.
    
    Input: path of file which contans document details for each query
    Output: List of List of score of each query-doc pair which contains [url-score, title-score, header-score, body-score, anchor-score]
    
'''
def make_feature(path, idf_score_dictionary, y_train_dictionary_for_each_query):
    all_query = []
    query_url_dictionary = {}
    X_dictionary_list_for_each_query = {}
    feature_list_of_all_documents = []
    pa3_signal_train_file = open(path)

    # List of All Queries
    pa3_signal_train_queries = pa3_signal_train_file.read().split("query: ")[1:]
    pa3_signal_train_file.close()

    final_feautre_list = []
    final_Y_list = []

    query_count = 0
    for query_single in pa3_signal_train_queries:

        all_query.append((query_single.split("\n"))[0])

        # List of All Url for a Query
        query_document_list = []

        X_all_document_for_a_query_list = []

        query_list = (query_single.split("\n  url: ")[0]).split(" ")
        documents_all = query_single.split("\n  url: ")[1:]

        each_query_feautre_list = []
        total_number_of_docs_in_query = len(documents_all)

        # All Document
        for document_single in documents_all:
            try:
                temp = document_single.split("\n")
            except:
                temp = []

            # URL
            url_score = 0
            try:
                url = temp[0]
                query_document_list.append(url)

                url_list = re.findall(r"[\w']+", url)
                dictionary_url = Counter(url_list)

                for query_term in query_list:
                    if query_term in list(dictionary_url.keys()):
                        url_score += (float(idf_score_dictionary[query_term])) * (float(dictionary_url[query_term]))
            except:
                url = []


            # Title Socre
            title_score = 0
            try:
                title_list = (str((temp[1].split("    title: "))[1])).split(" ")

                dictionary_title = Counter(title_list)

                for query_term in query_list:
                    if query_term in dictionary_title.keys():
                        title_score += (float(idf_score_dictionary[query_term])) * (float(dictionary_title[query_term]))
            except:
                title_list = []

            # Header Score
            header_score = 0
            try:
                start = document_single.find('    header: ')
                end = document_single.find('    body_hits: ', start)
                header_list = (
                str(("".join(((document_single[start:end]).split("    header: "))[1:])).replace("\n", " "))).split(" ")
                header_list.pop()
                dictionary_header = Counter(header_list)
                for query_term in query_list:
                    if query_term in dictionary_header.keys():
                        header_score += (float(idf_score_dictionary[query_term])) * (
                        float(dictionary_header[query_term]))
            except:
                header_list = []


            # Body Score
            body_score = 0
            try:
                body_list = ((document_single.split("\n    body_length: ")[0]).split("\n    body_hits: "))[1:]
                dictionary_body = {}
                for body_term in body_list:
                    body_term_postional_index = body_term.split(" ")
                    dictionary_body[body_term_postional_index[0]] = len(body_term_postional_index) - 1

                for query_term in query_list:
                    if query_term in dictionary_body.keys():
                        body_score += (float(idf_score_dictionary[query_term])) * (float(dictionary_body[query_term]))
            except:
                body = []

            # Anchor Score
            anchor_score = 0
            try:
                anchor_list = (document_single.split("    anchor_text: "))[1:]
                anchors = ((str("".join(anchor_list))).split("\n"))[0::2]
                dictionary_anchor = Counter((str(" ".join(anchors))).split(" "))

                for query_term in query_list:
                    if query_term in dictionary_anchor.keys():
                        anchor_score += (float(idf_score_dictionary[query_term])) * (
                        float(dictionary_anchor[query_term]))




            except:
                anchor_list = []

            score_list = [url_score, title_score, header_score, body_score, anchor_score]
            feature_list_of_all_documents.append(score_list)
            X_all_document_for_a_query_list.append(score_list)

        i = 0
        j = 0

        query_url_dictionary[query_count] = query_document_list
        X_dictionary_list_for_each_query[query_count] = X_all_document_for_a_query_list

        # Creating Training Data
        # Subtract two vector
        # Make Pair
        while i < total_number_of_docs_in_query:
            j = i + 1
            while j < total_number_of_docs_in_query:
                if y_train_dictionary_for_each_query[query_count][i] != y_train_dictionary_for_each_query[query_count][
                    j]:
                    final_feautre_list.append(
                        np.subtract(X_all_document_for_a_query_list[i], X_all_document_for_a_query_list[j]))
                    if y_train_dictionary_for_each_query[query_count][i] > \
                            y_train_dictionary_for_each_query[query_count][j]:
                        final_Y_list.append(1)
                    else:
                        final_Y_list.append(-1)

                j = j + 1
            i = i + 1

        query_count = query_count + 1

    return final_feautre_list, final_Y_list, feature_list_of_all_documents, X_dictionary_list_for_each_query, query_url_dictionary, all_query


'''
    Function
        Calculates idf-scores of each term.

    Input: Path of file which contains term-idf score pair
    Output: Dictionary of terms with key as term and value as idf-score

'''
def make_idf_weight_dictionary(path):
    term_score_dictionary = {}

    # Opening Idf Score Files And Storing Content
    file_idf_score = open(path)
    idf_score_all_terms = file_idf_score.read().split("\n")
    idf_score_all_terms.pop()
    file_idf_score.close()

    # Constructing IDF SCORE DICTIONARY
    for single_idf_term in idf_score_all_terms:
        temp = single_idf_term.split(":")
        term_score_dictionary[temp[0]] = temp[1]

    return term_score_dictionary


'''
    Function
        Get Relevance of each query-document pair.

    Input: Path of file which contains query document Relevance details.
    Output: List of relevance score of each query-document pair.
'''
def output_list_of_each_feautre(path):
    Y_dictionary_for_each_query = {}
    file_rel_doc_query = open(path)
    rel_doc_query = file_rel_doc_query.read()
    file_rel_doc_query.close()

    query_list_relevance = (rel_doc_query.split("query: "))[1:]

    query_count = 0
    for query in query_list_relevance:
        Y_list = []

        all_query_document_pair = query.split("  url: ")[1:]
        for single_document in all_query_document_pair:
            Y_list.append(float((str(single_document.split(" ")[1])).strip()))

        Y_dictionary_for_each_query[query_count] = Y_list
        query_count = query_count + 1

    return Y_dictionary_for_each_query


'''Function:from sklearn import preprocessing
    Input: Feautre Vector(List of List)
    Output: Scaled Vector having mean = 0 and standard deviation = 1(List of List)
'''
def scale_vector(feautre_list):
    return (preprocessing.scale(np.array(feautre_list)))



def write_data_to_file(path, feautres_list, output_class):
    file_to_write = open(path, "w")
    i = 0
    for feautre_vector in feautres_list:
        file_to_write.write(str(feautre_vector) + "\t" + str(output_class[i]) + "\n")
        i += 1
    file_to_write.close()
    return 0


def apply_svm(X, Y):
    clf = svm.SVC(probability=True, kernel='linear')
    clf.fit(X, Y)
    return clf


def calculate_ndcg_score_output_to_file(clf, X_test_dictionry_for_each_query, Y_each_query_test_dictionary,
                                        query_url_dictionary, query_list):
    if not os.path.exists("../output"):
        os.mkdir("../output")

    output_file = open("../output/predictedRelevance.txt", "w")

    ndcg_score = 0.0
    query_count = 0
    # ndcg = []

    for query in Y_each_query_test_dictionary.keys():

        output_file.write("queryNo: " + str(query_count + 1) + "\t" + "Query: " + query_list[query_count] + "\n")
        output_file.write("Ranking of URL:" + "\n")

        Y_test_predicted_value_list_for_each_query = clf.predict_proba((X_test_dictionry_for_each_query[query_count]))

        Y_test_predicted_value_list_for_each_query = [item[1] for item in Y_test_predicted_value_list_for_each_query]

        Y_ideal_value_list_for_each_query = Y_each_query_test_dictionary[query_count]

        # Ranking for predicted Value
        predicted_ranking_list = sorted(range(len(Y_test_predicted_value_list_for_each_query)),
                                        key=lambda k: Y_test_predicted_value_list_for_each_query[k])
        predicted_ranking_list.reverse()

        # Ranking for Ideal Document Relevance
        ideal_ranking_list = sorted(range(len(Y_ideal_value_list_for_each_query)),
                                    key=lambda k: Y_ideal_value_list_for_each_query[k])
        ideal_ranking_list.reverse()

        dcg = 0.0
        idcg = 0.0

        i = 1
        for document_index in ideal_ranking_list:
            output_file.write(
                query_url_dictionary[query_count][predicted_ranking_list[i - 1]] + "\t" + "PredictedValue: " + str(
                    Y_test_predicted_value_list_for_each_query[predicted_ranking_list[i - 1]]) + "\n")

            dcg += (
            (pow(2, Y_ideal_value_list_for_each_query[predicted_ranking_list[i - 1]]) - 1) / (log(i + 1, 2)))
            idcg += ((pow(2, Y_ideal_value_list_for_each_query[ideal_ranking_list[i - 1]]) - 1) / (log(i + 1, 2)))
            i += 1

        output_file.write("\n")

        if idcg != 0:
            ndcg_value_for_a_query = dcg / idcg
            ndcg_score += (ndcg_value_for_a_query)
            # ndcg.append(ndcg_value_for_a_query)

        query_count += 1

    output_file.write("NDCG SCORE:" + str(ndcg_score / query_count))
    output_file.close()
    return ndcg_score / query_count


def main():
    idf_score_dictionary = make_idf_weight_dictionary("../data/idfs.txt")

    Y_train_dictionary_for_each_query = output_list_of_each_feautre("../data/pa3.rel.train")
    X_train_list, Y_train_list, all_doc_train_vector, c, c, c = make_feature("../data/pa3.signal.train", idf_score_dictionary, Y_train_dictionary_for_each_query)

    X_train_list = scale_vector(X_train_list)

    clf = apply_svm(X_train_list, Y_train_list)

    Y_each_query_test_dictionary = output_list_of_each_feautre("../data/pa3.rel.dev")
    c, c, X_test_list, X_test_dictionry_for_each_query, query_url_dictionary, query_list = make_feature("../data/pa3.signal.dev", idf_score_dictionary, Y_each_query_test_dictionary)

    ndcg_score = calculate_ndcg_score_output_to_file(clf, X_test_dictionry_for_each_query, Y_each_query_test_dictionary, query_url_dictionary, query_list)
    print "NDCG Score:", ndcg_score

if __name__ == '__main__':
    main()
