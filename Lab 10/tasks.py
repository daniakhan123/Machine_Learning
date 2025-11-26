
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic clustering dataset
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
data = pd.DataFrame(X, columns=["Feature1", "Feature2"])

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 3. Find optimum number of clusters using Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 4. Apply K-means with chosen number of clusters (assume 4)
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Visualize clusters
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('K-means Clustering')
plt.show()
