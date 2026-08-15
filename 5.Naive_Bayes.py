import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

#help(GaussianNB)
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
###################### 7. Train_Test_Split #####################
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

###################### 8. Apply_NB #####################
model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

####################### 9. Result values calculation #####################
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

####################### 10. Draw Confusion Matrix #####################
ConfusionMatrixDisplay(cf).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
####################### 11. Draw Class Distribution #####################
df["Class"].value_counts().plot(kind ="bar")
plt.title("Class Distribution")
plt.xlabel("Bean Class")
plt.ylabel("Number of Samples")
plt.show()