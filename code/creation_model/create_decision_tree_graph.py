import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
#Models
from sklearn import tree
import pickle as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import graphviz
from sklearn.externals.six import StringIO
import pydot
import pydotplus





CodeSmell=["HAS","HBR","MIM","LIC","NLMR"]

for codesmell in CodeSmell:


    classification="Decision Tree"
    if (codesmell == "HAS"):
       print("--------HAS CodeSmell------------")
       path="../../dataset/HAS/taken/HAS_RandomUnderSampler.csv"
       file_name = "../Decision_Tree_graph/HAS_dot.pdf"

    if (codesmell == "HBR"):
        print("--------HBR CodeSmell------------")
        path = "../../dataset/HBR/taken/HBR_RandomUnderSampler.csv"
        file_name = "../Decision_Tree_graph/HBR_dot.pdf"

    if (codesmell == "MIM"):
        print("--------MIM CodeSmell------------")
        path = "../../dataset/MIM/taken/MIM_RandomUnderSampler.csv"
        file_name = "../Decision_Tree_graph/MIM_dot.pdf"

    if (codesmell == "LIC"):
        print("--------LIC CodeSmell------------")
        path = "../../dataset/LIC/taken/LIC_RandomUnderSampler.csv"
        clf = tree.DecisionTreeClassifier()
        equilibrage = "RandomUnderSampler"
        file_name = "../Decision_Tree_graph/LIC_dot.pdf"
    if (codesmell == "NLMR"):
        path = "../../dataset/NLMR/taken/NLMR_RandomUnderSampler.csv"
        equilibrage = "RandomUnderSampler"
        file_name="../Decision_Tree_graph/NLMR_dot.pdf"





    clf = tree.DecisionTreeClassifier()
    df = pd.read_csv(path)
    idx = np.random.permutation(df.index)
    df.reindex(idx)


    #Construction du dataset
    Y = df.is_code_smell.values
    df.drop('is_code_smell', axis=1, inplace=True)
    X = df.values










    #Construction du modele classificateur
    lr_model = clf.fit(X, Y)

    #Creation du graphe
    dot_data = tree.export_graphviz(clf, out_file=None,
            feature_names=df.columns,
            class_names=True,
            filled=True, rounded=True,
            special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(file_name)






























