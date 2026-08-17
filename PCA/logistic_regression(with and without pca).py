#1. Import section
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

#Check and remove null and duplicate values
df = pd.read_csv("WineQT.csv")
print(df.isnull().sum())
print(df.duplicated().sum())
#df = df.dropna()
df = df.fillna(df.mean(numeric_only = True))
df = df.drop_duplicates()

#Label Encoding
le = LabelEncoder()
for col in df.columns:
    if df[col].dtypes == "Object" or df[col].dtypes == "O":
        df[col] = le.fit_transform(df[col])

#Separate Features and Target
X = df.drop(columns=["quality"],axis = 1)
y = df["quality"]




#Class Distribution plot
values = df["quality"].value_counts()
values.plot(kind = "bar")
plt.xlabel("Bean Class")
plt.ylabel("Value Counts")
plt.title("Class Distribution before training")
plt.show()

#Train_Test_Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42
)

#print(y_train.head())
#Feature Scaling
scaling = StandardScaler()
X_train_scaled = scaling.fit_transform(X_train)
X_test_scaled = scaling.transform(X_test)

#Train the logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred, average = "weighted",zero_division = 0)
recall = recall_score(y_test, y_pred,average = "weighted",zero_division = 0)
f1 = f1_score(y_test, y_pred, average = "weighted",zero_division = 0)
cr = classification_report(y_test, y_pred,zero_division = 0)
cm = confusion_matrix(y_test, y_pred)
befor = [accuracy, precision, recall, f1]

print("Before Applying PCA")
print(f"Accuracy: {accuracy : .2f}")
print(f"Precision: {precision : .2f}")
print(f"Recall: {recall : .2f}")
print(f"f1 score: {f1 : .2f}")
print(f"Classification Report\n {cr}")
print(f"Confusion Matrix: \n{cm}")

#help(ConfusionMatrixDisplay)
#Plot Confusion Matrix
ConfusionMatrixDisplay(cm).plot(cmap = "Blues")
plt.title("Confusion Matrix before pca")
plt.show()

#After Applying PCA
pca = PCA(
    n_components = 2,
    random_state = 42
)
X_trained_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#Train the logistic Regression
model = LogisticRegression()
model.fit(X_trained_pca, y_train)
y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred, average = "weighted",zero_division = 0)
recall = recall_score(y_test, y_pred,average = "weighted",zero_division = 0)
f1 = f1_score(y_test, y_pred, average = "weighted",zero_division = 0)
cr = classification_report(y_test, y_pred,zero_division = 0)
cm = confusion_matrix(y_test, y_pred)

after = [accuracy, precision, recall, f1]
print("After Applying PCA")
print(f"Accuracy: {accuracy : .2f}")
print(f"Precision: {precision : .2f}")
print(f"Recall: {recall : .2f}")
print(f"f1 score: {f1 : .2f}")
print(f"Classification Report\n {cr}")
print(f"Confusion Matrix: \n{cm}")

#Plot Confusion Matrix
ConfusionMatrixDisplay(cm).plot(cmap = "Blues")
plt.title("Confusion Matrix After pca")
plt.show()

#Comparision Results before and after pca
name = ["Accuray", "recall", "precision", "f1"]
x = np.arange(4)
plt.bar(x-0.2, befor, 0.4, label = "before_pca")
plt.bar(x+0.2,after,width = 0.4, label = "after_pca")
plt.xticks(x,name)
plt.legend()
plt.show()
