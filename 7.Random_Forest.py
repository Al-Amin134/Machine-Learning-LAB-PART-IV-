import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
#help(RandomForestClassifier)
##################### 1.load data ############################
df = pd.read_csv("Dry_Bean.csv")
print(df.head())
 
###################### 2.find null value and duplicate values #####################
#print(df.isnull().sum())
print(df.duplicated().sum())
###################### 3. Remove Null and Duplicated Value #####################
df = df.dropna()
df = df.drop_duplicates()
###################### 4. Label Encoder #####################
le = LabelEncoder()
for col in df.columns:
    if df[col].dtypes == "object" or df[col].dtypes == "O":
        df[col] = le.fit_transform(df[col])
print(df.head())

###################### 5. Separate Features and Target #####################
X = df.drop(columns = "Class")
y = df["Class"]
#print(X.head())

#print(X_scaled[:5])
###################### 6. Train_Test_Split #####################
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

###################### 7. Find the best n #####################
values = range(50,450,50)
scores = []
for n in values:
    rf = RandomForestClassifier(
        n_estimators = n,
        random_state = 42,
        oob_score = True
    )
    rf.fit(
        X_train,
        y_train
    )
    oob_error = 1-rf.oob_score_
    scores.append(oob_error)
best_n = values[np.argmin(scores)]

###################### 8. Draw the graph to find n_estimator #####################
plt.plot(values,scores, marker="o")
plt.xlabel("Number of trees")
plt.ylabel("OOB Error")
print(f"The best vlaue for tree {best_n}")
plt.show()

#No need StandardScaler and PCA
###################### 9. Apply Decision Tree & Predict the target #####################
model = RandomForestClassifier(
    n_estimators = best_n,
    random_state = 42
)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

####################### 10. Result values calculation #####################
accuracy = accuracy_score(y_test, y_pred)
cf = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)
f1_value = f1_score(y_test, y_pred, average = "weighted")
recall_value = recall_score(y_test, y_pred, average = "weighted")
precision_value = precision_score(y_test, y_pred, average = "weighted")

print(f"Accuracy of the model is : {accuracy: .2f}")
print(f"F1-Score : {f1_value: .2f}")
print(f"Recall: {recall_value: .2f}")
print(f"Precision: {precision_value: .2f}")
print(f"Confusion_Matrix: {cf}")
print(f"Classification Report : {cr}")

####################### 11. Draw Confusion Matrix #####################
ConfusionMatrixDisplay(cf).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
####################### 12. Feature Importance #####################
importance = model.feature_importances_
plt.bar(X.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
