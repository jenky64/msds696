import csv
import numpy as np
import pandas as pd
import random

from matplotlib import pyplot as plt
import seaborn as sns

# gensim imports
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity
from gensim.models import KeyedVectors



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, multilabel_confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay


def calculate_similarities_count(data: list) -> list:
    """
    calculate cosine similarities between
    documents based on word count
    :param data: list of documents to process
    :return: nparray of similarities
    """
    vec = CountVectorizer()
    vec.fit(data)
    vec_matrix = vec.transform(data)
    cosine_sim = cosine_similarity(vec_matrix, vec_matrix)

    return cosine_sim


def calculate_similarities_tfidf(data: list) -> list:
    """
    calculate cosine similarities between
    documents based on tfidf calculation
    :param data: list of documents to process
    :return: nparray of similarities
    """
    tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
    tfidf_matrix = tfidf_vec.fit_transform(data)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def calculate_similarities_embed(gdf, vectors: bool = False):
    """
    calculate cosine similarities between
    documents base on word embedding
    :param gdf: dataframe of documents
    :return:
    """

    document_titles_list = gdf.identifier.tolist()   #story_titles_list
    document_list = gdf.text_unicode.tolist()

    sim_story_list: list = list()
    sim_score_list: list = list()
    sim_avg_list: list = list()
    score_avg_diff_list: list = list()

    if vectors:
        vec = KeyedVectors.load('authored.kv')
    else:
        vec = api.load("glove-wiki-gigaword-50")
    similarity_index = WordEmbeddingSimilarityIndex(vec) # this takes the word vectors from the model
    dictionary = Dictionary(document_list)
    tfidf = TfidfModel(dictionary=dictionary)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    print(f'starting enumerate')
    # iterate over each document and calculate the similarities
    # between that document and all other documents
    for num, document in enumerate(document_list):
        if num % 25 == 0:
            print(f'working on number: {num}')
        s0_tf = tfidf[dictionary.doc2bow(document)]
        index = SoftCosineSimilarity( tfidf[[dictionary.doc2bow(s) for s in document_list]], similarity_matrix )
        doc_similarity_scores = index[s0_tf]
        sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
        dss_list = doc_similarity_scores.tolist()
        si_list = sorted_indexes.tolist()

        idx2 = si_list[1]
        score = f'{dss_list[idx2]:0.4f}'

        # calculate average of similarity values
        sim_avg = f'{sum( sorted(dss_list)[:-1]) / (len(dss_list)-1):0.4f}'

        # append to corresponding lists
        sim_story_list.append(document_titles_list[idx2])
        sim_score_list.append(score)
        sim_avg_list.append(sim_avg)

    # now get difference between score and average
    sim_score_list = [float(x) for x in sim_score_list]
    sim_avg_list = [float(x) for x in sim_avg_list]
    score_avg_diff_list = (np.array(sim_score_list) - np.array(sim_avg_list)).tolist()

    sim_df = pd.DataFrame(list(zip(document_titles_list, sim_story_list, sim_score_list, sim_avg_list, score_avg_diff_list)),
                          columns=['case', 'most_similar', 'similarity_score', 'avg_similarity', 'score_avg_difference'])

    return sim_df


def most_similar(documents: list, sim_array: list, num_highest: int = 3):
    """
    iterate over the cosine similarity data and
    for each document find the one that is most
    similar. This is determined by finding the
    document with the highest cosine similarity
    value.
    we'll also get the average similarity and
    the difference between highest and average
    :param titles: title of stores
    :param sim_array: cosine similarity array
    :return:
    """

    identifier_list: list = list()
    sim_score_list: list = list()
    sim_justice_list: list = list()

    for idx, idx_array in enumerate(sim_array):
        if idx > 34:
            break
        most_similar_idx = list()
        most_similar_value = list()
        count = 0

        idx_list = idx_array.tolist()
        idx_list_sorted = sorted(idx_list, reverse = True)
        for x in idx_list_sorted:
            idx_list_index = idx_list.index(x)
            if idx_list_index > 34:
                most_similar_idx.append(idx_list_index)
                most_similar_value.append(x)
                count += 1
                if count == num_highest:
                    break

        for justice_id, sim_score in list(zip(most_similar_idx, most_similar_value)):
            identifier_list.append(documents[idx])
            sim_justice_list.append(documents[justice_id])
            sim_score_list.append(sim_score)


    sim_df = pd.DataFrame(list(zip(identifier_list, sim_justice_list, sim_score_list)),
                          columns=['identifier', 'most_similar_justice', 'similarity_score'])

    return sim_df


def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    return df_classification_report


def run_n_times(X,y,model, runs=50, params=None):
    highest_accuracy_score = 0
    accuracy_score_list = list()
    best_model = None
    models_dict = dict()
    model_sets = tuple()

    # run the basic classifier 100 times
    for i in range(runs):
        if model == 'LinearSVC':
            clf = LinearSVC(max_iter = 100000, dual=False)
        elif model == 'RandomForest':
             if params:
                 clf = RandomForestClassifier(**params)
             else:
                clf = RandomForestClassifier(n_estimators= 100)
        elif model == 'MultinomialNB':
            clf = MultinomialNB()
        elif model == 'GradientBoosting':
            clf = GradientBoostingClassifier()
        elif model == 'LDA':
            clf = LinearDiscriminantAnalysis()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, stratify = y)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        cur_accuracy_score = accuracy_score(y_test,predict)
        accuracy_score_list.append(cur_accuracy_score)
        models_dict[i] = dict()
        models_dict[i]['model'] = clf
        models_dict[i]['tt_split'] = (X_train, X_test, y_train, y_test)

        # after each iteration, get the best model as measured by accuracy
        if cur_accuracy_score > highest_accuracy_score:
            best_model = clf
            highest_accuracy_score = cur_accuracy_score
            model_sets = (X_train, X_test, y_train, y_test)

    # finally, print some useful statistics
    print(f'AVERAGE ACCURACY: {round(np.average(accuracy_score_list),4)}')
    print(f'ACCURACY STD: {round(np.std(accuracy_score_list),4)}')
    print(f'HIGHEST ACCURACY: {round(float(highest_accuracy_score),4)}')

    return highest_accuracy_score, accuracy_score_list, best_model, model_sets, models_dict




def run_100(X,y,model):
    highest_accuracy_score = 0
    accuracy_score_list = list()
    best_model = None
    models_dict = dict()
    model_sets = tuple()

    # run the basic classifier 100 times
    for i in range(100):
        if model == 'LinearSVC':
            clf = LinearSVC(max_iter = 100000, dual=False)
        elif model == 'RandomForest':
            clf = RandomForestClassifier(n_estimators= 150)
        elif model == 'MultinomialNB':
            clf = MultinomialNB()
        elif model == 'GradientBoosting':
            clf = GradientBoostingClassifier()
        elif model == 'LDA':
            clf = LinearDiscriminantAnalysis()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, stratify = y)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        cur_accuracy_score = accuracy_score(y_test,predict)
        accuracy_score_list.append(cur_accuracy_score)
        models_dict[i] = dict()
        models_dict[i]['model'] = clf
        models_dict[i]['tt_split'] = (X_train, X_test, y_train, y_test)

        # after each iteration, get the best model as measured by accuracy
        if cur_accuracy_score > highest_accuracy_score:
            best_model = clf
            highest_accuracy_score = cur_accuracy_score
            model_sets = (X_train, X_test, y_train, y_test)

    # finally, print some useful statistics
    print(f'AVERAGE ACCURACY: {round(np.average(accuracy_score_list),4)}')
    print(f'ACCURACY STD: {round(np.std(accuracy_score_list),4)}')
    print(f'HIGHEST ACCURACY: {round(float(highest_accuracy_score),4)}')

    return highest_accuracy_score, accuracy_score_list, best_model, model_sets, models_dict


def chart_results(value_list):
    # now let's view the model accuracy distribution
    accuracy_df = pd.DataFrame(value_list, columns=['accuracy_score'])
    x_val = [round(x, 4) for x in accuracy_df.accuracy_score.value_counts().index.tolist()]
    plt.rcParams["figure.figsize"] = [10,5]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.barplot(x=x_val, y=accuracy_df.accuracy_score.value_counts())

    ax.set_xlabel('Accuracy %',fontsize=16)
    ax.set_ylabel('Count',fontsize=16)
    ax.set_title('Accuracy Distribution',fontsize=18)

    plt.xticks(rotation=75)
    plt.yticks(size=13)
    plt.xticks(size=13)
    plt.show()


def chart_per_curiams(df, model_name, accuracy_type=None):
    tmp_df = df.loc[df['Classifier'] == model_name]
    tmp_df = tmp_df.transpose()

    tmp_df.drop('Classifier', inplace=True)
    tmp_df.reset_index(inplace=True)
    tmp_df = tmp_df.rename(columns = {'index':'justice', 0: 'count'})
    tmp_df = tmp_df.sort_values(by='count', ascending=False)
    tmp_df = tmp_df[tmp_df['count'] > 0]


    sns.set(color_codes=True)
    plt.rcParams["figure.figsize"] = [10,6]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.barplot(y='count', x="justice", data = tmp_df)

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x()+0.25, p.get_height()+.5))

    ax.set_xlabel('Justice',fontsize=18)
    ax.set_ylabel('predicted count',fontsize=18)
    ax.set_title(f'Predicted Curiam Author Counts: {accuracy_type} {model_name} Model',fontsize=20)

    plt.xticks(rotation=60, size=15)
    plt.yticks(size=14)
    plt.show()


def chart_per_curiamss(df, model_name, accuracy_type=None):
    tmp_df = df.loc[df['Classifier'] == model_name]
    tmp_df = tmp_df.transpose()

    tmp_df.drop('Classifier', inplace=True)
    tmp_df.reset_index(inplace=True)
    tmp_df = tmp_df.rename(columns = {'index':'justice', 0: 'count'})
    tmp_df = tmp_df.sort_values(by='count', ascending=False)
    tmp_df = tmp_df[tmp_df['count'] > 0]


    sns.set(color_codes=True)
    plt.rcParams["figure.figsize"] = [10,6]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.barplot(y='count', x="justice", data = tmp_df)

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x()+0.25, p.get_height()+.5))

    ax.set_xlabel('Justice',fontsize=18)
    ax.set_ylabel('predicted count',fontsize=18)
    ax.set_title(f'Predicted Curiam Author Counts: {accuracy_type} {model_name} Model',fontsize=20)

    plt.xticks(rotation=60, size=15)
    plt.yticks(size=14)
    plt.show()

CX = vec.transform(curiam_new_df.text)



def predict_per_curiam_text(curiam_df, curiam_transformed_data, model, model_name, justice_names, justices_map):
    tmp_justice_dict = dict()
    value_list = [model_name]

    for justice_name in sorted(justice_names):
        tmp_justice_dict[justice_name] = 0

    classification_columns = ['Classifier']
    classification_columns.extend(sorted(justice_names))
    tmp_classification_df = pd.DataFrame(columns=classification_columns)

    predict = model.predict(curiam_transformed_data)
    model_predicted_df = pd.DataFrame()
    model_predicted_df['case'] = curiam_df['case']
    model_predicted_df['justice'] = [justices_map[x] for x in predict]

    pc_series = model_predicted_df.justice.value_counts()
    pc_names = pc_series.index.tolist()
    pc_values = pc_series.values.tolist()
    for j_name, j_count in list(zip(pc_names, pc_values)):
        tmp_justice_dict[j_name] = j_count

    for justice in sorted(tmp_justice_dict.keys()):
        value_list.append(tmp_justice_dict[justice])

    tmp_df = pd.DataFrame([value_list], columns=classification_columns)

    return tmp_df




def predict_per_curiams(curiam_df, predict_df, model, model_name, justice_names, justices_map):

    tmp_justice_dict = dict()
    value_list = [model_name]

    for justice_name in sorted(justice_names):
        tmp_justice_dict[justice_name] = 0

    classification_columns = ['Classifier']
    classification_columns.extend(sorted(justice_names))
    tmp_classification_df = pd.DataFrame(columns=classification_columns)

    predict = model.predict(predict_df)
    model_predicted_df = pd.DataFrame()
    model_predicted_df['case'] = curiam_df['case']
    model_predicted_df['justice'] = [justices_map[x] for x in predict]

    pc_series = model_predicted_df.justice.value_counts()
    pc_names = pc_series.index.tolist()
    pc_values = pc_series.values.tolist()
    for j_name, j_count in list(zip(pc_names, pc_values)):
        tmp_justice_dict[j_name] = j_count

    for justice in sorted(tmp_justice_dict.keys()):
        value_list.append(tmp_justice_dict[justice])

    tmp_df = pd.DataFrame([value_list], columns=classification_columns)

    return tmp_df


def run_cross_validation(X, y, model, splits=10, repeats=50, params=None):

    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=57)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f'cross validation Training score mean: {round(scores.mean(),4)}')
    print(f'cross validation Training score std: {round(scores.std(),4)}')

    return scores


# simple function for output display
def run_and_display_model_values(X, y, splits, model='rf', repeats=10, params=None):

    if model == 'rf':
        if params:
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestClassifier()

    if model == 'svm':
        if params:
            model = LinearSVC(max_iter = 100000, dual=False)
        else:
            model = LinearSVC(max_iter = 100000, dual=False)

    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state = 57)
    scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs=-1)
    print(f'cross validation Training score mean: {round(scores.mean(),4)}')
    print(f'cross validation Training score std: {round(scores.std(),4)}')


    return scores

# simple function to get closest value to item in list
def closest_value(lst, K):
    '''https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/'''
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]


def get_mean_accuracy_model(accuracy_list, model_dict):
    # now we get the index of the closest value to the average
    closest = closest_value(accuracy_list, np.average(accuracy_list))
    closest
    idx = accuracy_list.index(closest)
    print(f'closest accuracy score: {accuracy_list[idx]}')
    print(f'index of closest accuracy score: {idx}')
    print(f'closest accuracy score found {accuracy_list.count(closest)} times')

    values = np.array(accuracy_list)
    indexes = np.where(values == closest)[0]
    print(f'indexes: {indexes}')

    idx = random.choice(indexes)
    print(f'choosing model from index: {idx}')
    avg_model = model_dict[idx]['model']

    return avg_model


def get_best_accuracy_model(accuracy_list, model_dict):
    # now we get the index of the best model
    values = np.array(accuracy_list)
    max_accuracy = np.max(accuracy_list)
    indexes = np.where(values == max_accuracy)[0]
    print(f'indexes: {indexes}')

    #choose random index value and get corresonding best model
    idx = random.choice(indexes)
    print(f'max accuracy: {idx}')
    print(f'choosing model from index: {max_accuracy}')
    best_model = model_dict[idx]['model']

    return best_model


def add_names(df, names):
    current_names = df.justice.unique()
    to_add = list(set(names) - set(df.justice.unique()))
    for name in to_add:
        df = df.append({'justice': name, 'case': 0}, ignore_index=True)

    df = df.sort_values(by=['justice'])
    df = df.reset_index(drop=True)

    return df