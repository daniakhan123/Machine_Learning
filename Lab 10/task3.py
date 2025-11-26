import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load financial data
data = pd.read_csv("financial_data.csv")  # Make sure this file exists

# 2. Explore dataset
print(data.head())
print(data.describe())

# 3. Select relevant features
features = data[['Income', 'Spending', 'Savings', 'Investment']]  # Replace with actual columns

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 5. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 6. Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)

# Plot cumulative explained variance
plt.figure(figsize=(6,4))
plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.show()

# 7. Reduce dimensionality (e.g., retain 2 components)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
reduced_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])

# 8. Visualize reduced data
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Financial Data PCA')
plt.show()
