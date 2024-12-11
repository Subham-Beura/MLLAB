from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='r', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering')
plt.legend()
plt.show()
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg}")


from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
X_scaled.T,
c=n_clusters,
m=2.0,
error=0.005,
maxiter=1000,
init=None,
seed=42
)
fcm_labels = np.argmax(u, axis=0)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
plt.scatter(X_scaled[fcm_labels == i, 0], X_scaled[fcm_labels == i, 1], label=f'Cluster {i + 1}')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()
fcm_partition_coefficient = np.sum(u**2) / len(X_scaled)
print(f"Fuzzy Partition Coefficient (PC): {fcm_partition_coefficient:.2f}")
