#!/usr/bin/env python
# coding: utf-8

# IMPORT DEPENDENCIES
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold




# DATA AND GLOBAL VARIABLES
df = pd.read_csv('clientfinaldropped.csv').fillna(0)
models = [MultinomialNB, KNeighborsClassifier, RandomForestClassifier]

# Visualize the Class balance using a bar chart
st.title("Frequency of Deadbeats")
plt.hist(df['Type'], bins = 2, rwidth = 0.5,)
st.pyplot()

# DEFINE FUNCTIONS

def rescale_numbers(df, scaler):
    for col in df:
        if df[col].dtype in ['int64', 'float64']:
            numbers = df[col].astype(float).values.reshape(-1,1)
            df[col] = scaler().fit_transform(numbers)
    return df


def preprocess(df):
    return (df
           .pipe(rescale_numbers, MinMaxScaler)
           )


def train_test(df, target):
    return train_test_split(
    df[[col for col in df if col != target]],
    df[target],
    test_size = .2,
    random_state = 42
    )

def evaluate_model(algorithm, train_test):
    train_X, test_X, train_y, test_y = train_test
    model = algorithm().fit(train_X, train_y)
    pred_proba_y = model.predict_proba(test_X)
    auc = roc_auc_score(test_y, pred_proba_y[:, 1])
    st.subheader('Area under the Curve Score')
    st.write(auc)

    false_pos_rates, true_pos_rates, _ = roc_curve(test_y, pred_proba_y[:, 1])
    st.subheader('Area under the Curve Graph for this KFold Sample')
    plt.plot(false_pos_rates, true_pos_rates)
    st.pyplot()

    score = model.score(test_X, test_y)
    st.write(f"Accuracy: {round(score, 2)}")
    return model , score


def k_fold(df, target):
    features = df[[col for col in df if col != target]]
    target = df[target]
    kf = StratifiedKFold(n_splits = 5, random_state = 42)
    for model in models:
        st.title(model)
        for train_i, test_i in kf.split(features, target): 
            scores = []
            scores.append(
                evaluate_model(
                    model,
                    (features.iloc[train_i],
                     features.iloc[test_i],
                     target.iloc[train_i],
                     target.iloc[test_i])
                    )[1])   
        st.title("Average Model Score")
        st.write(sum(scores) / len(scores))
        
            
# call the functions
k_fold(preprocess(df), target = 'Type')



