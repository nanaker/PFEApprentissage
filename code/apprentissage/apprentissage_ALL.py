import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
#Models
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import graphviz
from sklearn.externals.six import StringIO
import pydot
from openpyxl import Workbook



CodeSmell=["HAS","HBR","MIM","LIC","NLMR"]



result = pd.DataFrame(columns=['defaut_de_code','classification_method', 'undersimpling_method','precision','rappel','F_mesure','Accuracy','Error Rate','Specificity','False Positive Rate ','False Negative Rate'])
print(result)


for codesmell in CodeSmell:


    classification="Decision Tree"
    if (codesmell == "HAS"):
       print("--------HAS CodeSmell------------")
       path="../../dataset/HAS/taken/HAS_NearMiss.csv"
       clf = GaussianNB()
       equilibrage="InstanceHardnessThreshold"
    if (codesmell == "HBR"):
        print("--------HBR CodeSmell------------")
        path = "../../dataset/HBR/taken/HBR_RandomUnderSampler.csv"
        clf = tree.DecisionTreeClassifier()
        equilibrage = "InstanceHardnessThreshold"
    if (codesmell == "MIM"):
        print("--------MIM CodeSmell------------")
        path = "../../dataset/MIM/taken/MIM_RandomUnderSampler.csv"
        clf = tree.DecisionTreeClassifier()
        equilibrage = "RandomUnderSampler"
    if (codesmell == "LIC"):
        print("--------LIC CodeSmell------------")
        path = "../../dataset/LIC/taken/LIC_RandomUnderSampler.csv"
        clf = tree.DecisionTreeClassifier()
        equilibrage = "RandomUnderSampler"
    if (codesmell == "NLMR"):
        path = "../../dataset/NLMR/taken/NLMR_RandomUnderSampler.csv"
        clf = svm.SVC(gamma='scale')
        equilibrage = "RandomUnderSampler"

    print("\n\n")
    print(path)
    print("\n\n")
    df = pd.read_csv(path)


    print("\n----------Using cross Validation------------------------------------")
    test_methode = "cross Validation"
    kf = KFold(n_splits=5, shuffle=True)


    F_mesures=[]
    Precisionss = []
    Rappelss = []
    Accuracyss=[]
    ErrorRatess=[]
    Specifityss=[]
    FalsePositiveRatess=[]
    FalseNegativeRatess=[]

    kfold = KFold(5, True, 1)
    df = pd.read_csv(path)

    # Construction du dataset
    Y = df.is_code_smell.values
    df.drop('is_code_smell', axis=1, inplace=True)
    X = df.values

    k=1


    for train, test in kfold.split(X):
      try :
       print("\n----------k=", k, "------------------------------------\n")
       k=k+1
       x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]

       model = clf.fit(x_train, y_train)
       predictions = model.predict(x_test)
       cnf_matrix = confusion_matrix(y_test, predictions)
       TP = cnf_matrix[1, 1]
       TN = cnf_matrix[0, 0]
       FP = cnf_matrix[0, 1]
       FN = cnf_matrix[1, 0]
       Precision = TP / (TP + FP)
       Rappel = TP / (TP + FN)
       F_mesure = 2 * Rappel * Precision / (Precision + Rappel)
       Accuracy=(TP + TN) / (TP + TN + FP + FN)
       ErrorRate=1-Accuracy
       Specificity = TN / (TN + FP)
       FalsePositiveRate=FP/(FP+TN)
       FalseNegativeRate=FN/(FN+TP)

       print("F_Mesure=", F_mesure)
       Precisionss.append(Precision)
       Rappelss.append(Rappel)
       F_mesures.append(F_mesure)
       Accuracyss.append(Accuracy)
       ErrorRatess.append(ErrorRate)
       Specifityss.append(Specificity)
       FalsePositiveRatess.append(FalsePositiveRate)
       FalseNegativeRatess.append(FalseNegativeRate)
      except:
          pass

    print("F_Mesures moyenne =", np.mean(F_mesures))

    result=result.append(pd.Series([codesmell,classification,equilibrage,np.mean(Precisionss),np.mean(Rappelss),np.mean(F_mesures),np.mean(Accuracyss),np.mean(ErrorRatess),np.mean(Specifityss),np.mean(FalsePositiveRatess),np.mean(FalseNegativeRatess)], index=result.columns), ignore_index=True)


result.to_excel('../../dataset/ALL_train_result_test.xlsx', index=False)





























