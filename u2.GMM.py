import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
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

#use BIC method to find the best k
values = range(1,10)
bic_scores = []
for k in values:
    gmm = GaussianMixture(
        n_components = k,
        random_state = 42
    )
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
plt.plot(values, bic_scores, marker = "o")
plt.show()

#best_k = 5
best_k  = values[np.argmin(bic_scores)]
print(f"The best value of k is : {best_k}")
#Train the model
gmm = GaussianMixture(
    n_components = k,
    random_state = 42
)
gmm.fit(X_scaled)
clusters = gmm.predict(X_scaled)

#Calculate the Silhouette Score
s_score = silhouette_score(
    X_scaled, clusters
)
#to find NMI and ARI we need true label but the dataset down not have true label
print(f"The Silhouet Score is: {s_score: .2f} ")
#Find the Center of each of the clusters
center = gmm.means_
center = scaling.inverse_transform(center)

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c = clusters,
    s = 200
)
plt.scatter(
    center[:,0],
    center[:,1],
    marker = "x",
    color = "r",
    s = 200
)
plt.title("Gaussian Mixture Model")
plt.show()

