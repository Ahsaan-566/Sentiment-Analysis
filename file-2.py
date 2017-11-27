from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
from sklearn.model_selection import train_test_split
from normalization import normalize_corpus
from features import bow_extractor, tfidf_extractor
from features import averaged_word_vectorizer
from features import tfidf_weighted_averaged_word_vectorizer
import nltk
from gensim import models
import gensim
from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import pandas as pd

dataset = pd.read_csv("/home/drogon/Desktop/Data mining/Text mining/Trainset.csv", delimiter=',', encoding="latin-1")


# print list(dataset.columns.values)
# print  dataset['rating']
# print dataset['review']
#print dataset


def prepare_datasets(corpus, labels, test_data_proportion=0.0):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, test_size=0.0, random_state=40)
    return train_X, test_X, train_Y, test_Y

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels

#dataset = get_data()

labels = dataset['rating']
corpus = dataset['review']


corpus, labels = remove_empty_docs(corpus, labels)


train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus, labels, test_data_proportion = 0.0)

norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus)

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
#bow_test_features = bow_vectorizer.transform(test_corpus)

# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)

# tokenize documents
tokenized_train = [nltk.word_tokenize(text)
                   for text in train_corpus]
tokenized_test = [nltk.word_tokenize(text)
                 for text in test_corpus]

# build word2vec model
model = gensim.models.Word2Vec(tokenized_train,
                               size=500,
                               window=100,
                               min_count=30,
                               sample=1e-3)

# averaged word vector features
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,
                                                 model=model,
                                                 num_features=500)

avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
                                                model=model,
                                                num_features=500)

# tfidf weighted averaged word vector features
vocab = tfidf_vectorizer.vocabulary_
tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train,
                                                                  tfidf_vectors=tfidf_train_features,
                                                                  tfidf_vocabulary=vocab,
                                                                  model=model,
                                                                  num_features=500)

tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,
                                                                 tfidf_vectors=tfidf_test_features,
                                                                 tfidf_vocabulary=vocab,
                                                                 model=model,
                                                                 num_features=500)

def get_metrics(true_labels, predicted_labels):
    print 'Accuracy:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        4)
    print 'Precision:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2)
    print 'Recall:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2)
    print 'F1 Score:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2)

def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):

    # build model
    classifier.fit(train_features, train_labels)

    # predict using model
    predictions = classifier.predict(test_features)

    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions


mnb = MultinomialNB()
svm_algo = SGDClassifier(loss='log' , max_iter=20000)
linear_svm = svm.LinearSVC(C=5.0, loss='hinge', max_iter=10000)


#1-linear svm with bag of words features
linear_svm_bow_predictions = train_predict_evaluate_model(classifier=linear_svm,
                                                          train_features=bow_train_features,
                                                          train_labels=train_labels,
                                                          test_features=bow_test_features,
                                                          test_labels=test_labels)

#2- Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm_algo,
                                                   train_features=bow_train_features,
                                                   train_labels=train_labels,
                                                   test_features=bow_test_features,
                                                   test_labels=test_labels)

# 3- Support Vector Machine with tfidf_features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm_algo,
                                                     train_features=tfidf_train_features,
                                                     train_labels=train_labels,
                                                     test_features=tfidf_test_features,
                                                     test_labels=test_labels)

#4- Linear SVM with tfidf features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=linear_svm,
                                                     train_features=tfidf_train_features,
                                                     train_labels=train_labels,
                                                     test_features=tfidf_test_features,
                                                     test_labels=test_labels)


#5- Support Vector Machine with averaged word vector features
svm_avgwv_predictions = train_predict_evaluate_model(classifier=svm_algo,
                                                     train_features=avg_wv_train_features,
                                                     train_labels=train_labels,
                                                     test_features=avg_wv_test_features,
                                                     test_labels=test_labels)


#6- Support Vector Machine with tfidf weighted averaged word vector features
svm_tfidfwv_predictions = train_predict_evaluate_model(classifier=svm_algo,
                                                       train_features=tfidf_wv_train_features,
                                                       train_labels=train_labels,
                                                       test_features=tfidf_wv_test_features,
                                                       test_labels=test_labels)

#7- Multinomial Naive Bayes with tfidf features
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                     train_features=tfidf_train_features,
                                                     train_labels=train_labels,
                                                     test_features=tfidf_test_features,
                                                     test_labels=test_labels)

#8- Multinomial Naive Bayes with bag of words features
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb, train_features=bow_train_features,
                                                   train_labels=train_labels,
                                                   test_features=bow_test_features,
                                                   test_labels=test_labels)



# cm = metrics.confusion_matrix(test_labels, svm_tfidf_predictions)
# pd.DataFrame(cm, index=range(0,20), columns=range(0,20))

################################## Testing Kaggle #########################################

test_set = pd.read_csv("/home/drogon/Desktop/Data mining/Text mining/Testset.csv", delimiter=',',encoding='latin-1')


# def remove_empty_docs_test(corpus, labels):
#     filtered_corpus = []
#     filtered_labels = []
#     for doc, label in zip(corpus, labels):
#         if doc.strip():
#             filtered_corpus.append(doc)
#             filtered_labels.append(label)
#     return filtered_corpus, filtered_labels

#dataset = get_data()

corpus_test = test_set['review']

#corpus_test= remove_empty_docs(corpus_test)
#norm_test = normalize_corpus(corpus_test)

# bag of words features
bow_test_features = bow_vectorizer.transform(corpus_test)

# tfidf features
tfidf_test_features = tfidf_vectorizer.transform(corpus_test)

# tokenize documents
tokenized_test = [nltk.word_tokenize(text)
                   for text in corpus_test]

# # averaged word vector features
# avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
#                                                 model=model,
#                                                 num_features=500)

# tfidf weighted averaged word vector features
# vocab = tfidf_vectorizer.vocabulary_
# tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,
#                                                                  tfidf_vectors=tfidf_test_features,
#                                                                  tfidf_vocabulary=vocab,
#                                                                  model=model,
#                                                                  num_features=500)

# linear_svm.fit(tfidf_train_features, train_labels)
# output = linear_svm.predict(tfidf_test_features)
# prediction = pd.DataFrame({'id': test_set['id'],'rating': output})
# prediction.to_csv('kaggle_python(1-5)-no-normalize.csv', index=False)

svm_algo.fit(tfidf_train_features,train_labels)
out = svm_algo.predict(tfidf_test_features)
prediction = pd.DataFrame({'id': test_set['id'],'rating': out})
prediction.to_csv('kaggle_python-final2.csv', index=False)




