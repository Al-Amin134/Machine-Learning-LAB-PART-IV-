import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
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

dbscan = DBSCAN(
    eps = 0.5,
    min_samples = 10
)

dbscan.fit(X_scaled)
clusters = dbscan.labels_
#print("Done")

#Find the number of the clusters
unique_clusters = set(clusters)
cluster_counts = len(unique_clusters)
if -1 in unique_clusters:
    cluster_counts = cluster_counts - 1
print(f"Total cluster is: {cluster_counts}")

# Differential clusters and plot them
mask = clusters!=-1
masked_x = X_scaled[mask]

masked_cluster = clusters[mask]

if cluster_counts>1:
    score = silhouette_score(
        masked_x,
        masked_cluster
    )
print(f"The silhouette score is : {score: .2f}")

#Plot Cluster
simple = clusters!=-1
noise = clusters==-1
plt.scatter(
    X.iloc[simple,0],
    X.iloc[simple,1],
    c = clusters[simple],
    s =100
)

plt.scatter(
    X.iloc[noise,0],
    X.iloc[noise,1],
    color = "r",
    s =100
    
)
plt.title("DB SCAN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
#plot noise points

plt.show()
