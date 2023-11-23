import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

events = {"IN": "100001", "OUT": "010010", }
result = {"confusion": [], "accuracy": 0, "precise": 0, "recall": 0}


def Seperate(data):
    """seperate sensors data and observations"""
    df1 = data.query("(Return!='IN'& Return!='OUT')")
    df2 = data.query("(Return=='IN'| Return=='OUT')")
    df = pd.merge(df1, df2, how='left', on='Time')
    df.columns = ["Time", "Return", "Obser"]
    return df


def Frequency(data, fre):
    """Get data in different frequence from raw data"""
    df = copy.deepcopy(data)
    index = df[df["Obser"].isnull() == False].index
    for i in index:
        for j in range(1, fre):
            df["Obser"].loc[i + j] = df["Obser"].loc[i]
    return df[::fre]


def Format(data):
    """Translate the data to result that can be compared in confusion matrix"""
    df = copy.deepcopy(data)
    df["Obser"].fillna("NO", inplace=True)
    for i in range(1, len(df) - 1):
        event = df["Return"].loc[i - 1] + df["Return"].loc[i] + df["Return"].loc[i + 1]
        if event == events["IN"]:
            df.loc[i, "Return"] = "IN"
        elif event == events["OUT"]:
            df.loc[i, "Return"] = "OUT"
    df["Return"].replace(["00", "10", "01", "11"], ["NO", "NO", "NO", "NO"], inplace=True)
    return df


def Accuracy(data):
    data = Format(data)
    y_true = np.array(data["Obser"])
    y_pred = np.array(data["Return"])
    # result["confusion"] = metrics.confusion_matrix(y_true, y_pred)
    result["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    result["precise"] = metrics.precision_score(y_true, y_pred,average='macro')
    result["recall"] = metrics.recall_score(y_true, y_pred,average='macro')
    return result


raw_data = pd.read_csv("collecting_data_F1D1.csv", names=["Time", "Return"], skiprows=2)
data = Seperate(raw_data)
data = Format(data)
print(Accuracy(data))

# result = []
# for i in range(1,7):
#     result.append(Accuracy(Frequency(data,i)))
# print(result)
