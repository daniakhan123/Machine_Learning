import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load the customer data
data = pd.read_csv("shopping_data.csv")  

# 2. Explore dataset
print(data.head())
print(data.describe())

# 3. Select relevant features
features = data[['Annual_Spend', 'Items_Purchased']]  #

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 5. Determine optimal number of clusters using Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Customer Segmentation')
plt.show()

# 6. Apply K-means with chosen number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 7. Visualize clusters
plt.figure(figsize=(6,4))
plt.scatter(data['Annual_Spend'], data['Items_Purchased'], c=data['Cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Spend')
plt.ylabel('Items Purchased')
plt.title('Customer Segments')
plt.show()

# 8. Analyze cluster centers
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Annual_Spend','Items_Purchased'])
print("Cluster centers:\n", centers)
