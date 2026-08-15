import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
#help(plot_tree)
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

#No need StandardScaler and PCA
###################### 7. Apply Decision Tree & Predict the target #####################
model = DecisionTreeClassifier(
    random_state = 42,
    max_depth = 5
)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

####################### 8. Result values calculation #####################
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


# You can ignore this part
####################### 9. Draw Confusion Matrix #####################
ConfusionMatrixDisplay(cf).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

####################### 10. plot tree #####################
plt.subplot(2,1,1)
plot_tree(
    model
)
plt.subplot(2,1,2)
####################### 11. Feature Importance #####################
importance = model.feature_importances_
plt.bar(X.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()