#1. Import section
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
#help(cross_val_score)

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
X_train= scaling.fit_transform(X_train)
X_test= scaling.transform(X_test)

#find the value of k by using the cross validation 
values = range(1,30)
scores = []
for k in values:
    knn = KNeighborsClassifier(
        n_neighbors = k
    )
    knn.fit(X_train,y_train)
    cross_val = cross_val_score(
        knn,
        X_train,
        y_train
    )
    scores.append(cross_val.mean())
best_k = values[np.argmax(scores)]
print(f"The best value of k : {best_k}")
plt.plot(values, scores, marker = "o")
plt.title("Find the best value curve")
plt.xlabel("Number of K")
plt.ylabel("Cross Validation Score")
plt.show()


#Train the KNN
knn = KNeighborsClassifier(
        n_neighbors = best_k
    )
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average = "weighted", zero_division = 0)
recall = recall_score(y_test, y_predict, average = "weighted", zero_division = 0)
f1 = f1_score(y_test, y_predict, average = "weighted", zero_division = 0)
cr = classification_report(y_test, y_predict, zero_division = 0)
cm = confusion_matrix(y_test, y_predict)

before = [accuracy, precision, recall, f1]

print("Before Applying PCA")
print(f"Accuracy: {accuracy : .2f}")
print(f"Precision: {precision : .2f}")
print(f"Recall: {recall : .2f}")
print(f"f1 score: {f1 : .2f}")
print(f"Classification Report\n {cr}")
print(f"Confusion Matrix: \n{cm}")

#help(ConfusionMatrixDisplay)
ConfusionMatrixDisplay(
    cm
).plot(cmap="Blues")
plt.title("Confusion matrix Before PCA")
plt.show()



#Apply PCA
pca = PCA(
    n_components = 2,
    random_state = 42
)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
knn = KNeighborsClassifier(
        n_neighbors = best_k
    )
knn.fit(X_train_pca, y_train)
y_predict = knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average = "weighted", zero_division = 0)
recall = recall_score(y_test, y_predict, average = "weighted", zero_division = 0)
f1 = f1_score(y_test, y_predict, average = "weighted", zero_division = 0)
cr = classification_report(y_test, y_predict, zero_division = 0)
cm = confusion_matrix(y_test, y_predict)

after = [accuracy, precision, recall, f1]

print("After Applying PCA")
print(f"Accuracy: {accuracy : .2f}")
print(f"Precision: {precision : .2f}")
print(f"Recall: {recall : .2f}")
print(f"f1 score: {f1 : .2f}")
print(f"Classification Report\n {cr}")
print(f"Confusion Matrix: \n{cm}")

ConfusionMatrixDisplay(
    cm
).plot(cmap="Blues")
plt.title("Confusion matrix After PCA")
plt.show()

name = ["accuracy", "precision", "recall", "f1"]
x = np.arange(4)
plt.bar(x-0.2, before, width = 0.4, label = "Before PCA")
plt.bar(x+0.2, after, width = 0.4, label = "After PCA")
plt.legend()
plt.xticks(x,name)
plt.ylabel("Scores")
plt.title("Comparision before and After PCA")
plt.show()