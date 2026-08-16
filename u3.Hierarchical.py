import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn import metrics

df = pd.read_csv("Mall_Customers.csv")
print(df.head())

print(df.isnull().sum())
print(df.duplicated().sum())

df = df.dropna()
df = df.drop_duplicates()

#df = df.drop("CustomerID",axis =1 )
df = df.drop(columns = ["CustomerID"])

#Encode features
le = LabelEncoder()
for col in df.columns :
    if df[col].dtypes=="Object" or df[col].dtypes =="O":
        df[col]=le.fit_transform(df[col])
print(df.head())

X = df[["Spending Score (1-100)","Annual Income (k$)"]]
print(X.head())

#Scaling
scaling = StandardScaler()
X_scaled = scaling.fit_transform(X)

#Dendogram Drawing
linked = linkage(
    X_scaled,
    method = "ward"
)

dendrogram(linked,truncate_mode='level', p = 5)
plt.title("Dendogram")
plt.xlabel("Simple Index")
plt.ylabel("Distance")
plt.show()

#Apply Agglomerative Clustering
model = AgglomerativeClustering(
    n_clusters = 5,
    linkage = "ward"                            
)
clusters = model.fit_predict(X_scaled)

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c = clusters,
    s = 100
    )
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
#Calculate the Silhouette Score

s_score = silhouette_score(
    X_scaled, clusters
)
#to find NMI and ARI we need true label but the dataset down not have true label
print(f"The Silhouet Score is: {s_score: .2f} ")
#Find the Center of each of the clusters

