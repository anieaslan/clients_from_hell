"""
This file is designed to be used with Streamlit. In Terminal, enter streamlit run project.py
Ani, Haimant, and Sam designed a Streamlit application to visualize a classification algorithm based around Clients From Hell.
The goal is to classify a potential client as a Deadbeat or not, depending on the word usage used in communications.
"""

# DEPENDENCIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
np.random.seed(42)
from math import log
import streamlit as st

# Scikit Learn DEPENDENCIES
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# DEFINE THE FUNCTIONS

# TERM FREQUENCY
def tf(bag_of_words):
    totals = bag_of_words.sum()
    num_words = totals.sum()
    return totals / num_words

# INVERSE DOCUMENT FREQUENCY
def idf(bag_of_words):
    num_docs = bag_of_words.shape[0]
    doc_freq = bag_of_words.notnull().sum()
    return (num_docs / doc_freq).apply(log)

# COMBINE THEM
def tf_idf(bag_of_words):
    return tf(bag_of_words) * idf(bag_of_words)


# IMPORT THE DATA
# try: 
df = pd.read_csv('clientfinaldropped.csv')
# except:
    # continue # add in the line where we use the scraper


# PREPARE THE DATA FOR MODEL TESTING
features = df[[col for col in df != 'Type']].fillna(0)
target = df['Type']


# run the actual raw data through each model, provide a score and roc_auc_score visualization for each
train_X, test_X, train_y, test_y = train_test_split(
	features,
	target,
	test_size  = 0.2,
	random_state = 42,
)


# RandomForest
rf = RandomForestClassifier(n_estimators = 100).fit(train_X, train_y)
predictions = rf.predict(test_X)
st.title('Random Forest')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = rf.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# LogisticRegression
logreg = LogisticRegression().fit(train_X, train_y)
predictions = logreg.predict(test_X)
st.title('Logistic Regression')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = logreg.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# KNearestNeighbors
knn = KNeighborsClassifier(n_neighbors = 5).fit(train_X, train_y.values.ravel())
predictions = knn.predict(test_X)
st.title('K-Nearest Neighbors')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = knn.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# Multinomial Naive Bayes
mnb = MultinomialNB().fit(train_X, train_y)
predictions = mnb.predict(test_X)
st.title('Multinomial Naive Bayes')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = mnb.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# minmaxscaler the data, run it through each model, provide a score and roc_auc_score visualization for each
minmax = MinMaxScaler().fit_transform(features)
train_X, test_X, train_y, test_y = train_test_split(
	minmax,
	target,
	test_size  = 0.2,
	random_state = 42,
)

# RandomForest
rf = RandomForestClassifier(n_estimators = 100).fit(train_X, train_y)
predictions = rf.predict(test_X)
st.title('Random Forest')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = rf.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# LogisticRegression
logreg = LogisticRegression().fit(train_X, train_y)
predictions = logreg.predict(test_X)
st.title('Logistic Regression')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = logreg.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# KNearestNeighbors
knn = KNeighborsClassifier(n_neighbors = 5).fit(train_X, train_y.values.ravel())
predictions = knn.predict(test_X)
st.title('K-Nearest Neighbors')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = knn.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# Multinomial Naive Bayes
mnb = MultinomialNB().fit(train_X, train_y)
predictions = mnb.predict(test_X)
st.title('Multinomial Naive Bayes')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = mnb.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# robust scaler the data, run it through each model, provide a score and roc_auc_score visualization for each
robust = RobustScaler().fit_transform(features)
train_X, test_X, train_y, test_y = train_test_split(
	robust,
	target,
	test_size  = 0.2,
	random_state = 42,
)

# RandomForest
rf = RandomForestClassifier(n_estimators = 100).fit(train_X, train_y)
predictions = rf.predict(test_X)
st.title('Random Forest')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = rf.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# LogisticRegression
logreg = LogisticRegression().fit(train_X, train_y)
predictions = logreg.predict(test_X)
st.title('Logistic Regression')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = logreg.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# KNearestNeighbors
knn = KNeighborsClassifier(n_neighbors = 5).fit(train_X, train_y.values.ravel())
predictions = knn.predict(test_X)
st.title('K-Nearest Neighbors')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = knn.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)


# Multinomial Naive Bayes
mnb = MultinomialNB().fit(train_X, train_y)
predictions = mnb.predict(test_X)
st.title('Multinomial Naive Bayes')
st.subheader('Model Score')
st.write(r2_score(predictions, test_y))
pred_proba_y = mnb.predict_proba(test_X)
# Area under the Curve visualization
false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Graph')
plt.plot(false_pos_rates, true_pos_rates)
st.pyplot()
# Area under the Curve percentage
auc = roc_auc_score(test_y, pred_proba_y[:, 1])
st.subheader('Area under the Curve Score')
st.write(auc)

