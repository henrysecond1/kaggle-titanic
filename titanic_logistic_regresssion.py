from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def load_train_data(filepath):
    features_to_select = ['Pclass', 'Sex', 'Age']
    df = pd.read_csv(filepath, usecols=['Survived'] + features_to_select)

    if 'Sex' in features_to_select:
        df['Sex'] = df['Sex'].map({'male': 1, 'female':0})
    if 'Age' in features_to_select:
        df['Age'] = df['Age'].fillna(df['Age'].mean())

    #print (tabulate(df, headers='keys', tablefmt='psql'))

    Y = df['Survived'].values
    X = df.drop('Survived',1).values
    
    return Y, X

def load_test_data(filepath):
    features_to_select = ['Pclass', 'Sex', 'Age']
    ids = pd.read_csv(filepath, usecols=['PassengerId'])
    df = pd.read_csv(filepath, usecols=features_to_select)

    if 'Sex' in features_to_select:
        df['Sex'] = df['Sex'].map({'male': 1, 'female':0})
    if 'Age' in features_to_select:
        df['Age'] = df['Age'].fillna(df['Age'].mean())

    #print (tabulate(df, headers='keys', tablefmt='psql'))

    X = df.values
    return ids, X

def train(Y, X, X_test, ids):
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(X, Y)

    Y_predict = model.predict(X_test)
    ids['Survived'] = Y_predict

    output_csv = ids.to_csv('result.csv', index=False, encoding='utf-8')
    
def main():
    Y, X = load_train_data('./train.csv')
    ids, X_test = load_test_data('./test.csv')

    train(Y, X, X_test, ids)

if __name__ == "__main__":
    main()
