import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


X = np.array([(1, 1), (1, 2), (2, 2), (3, 3), (7, 7), (8, 7), (8, 8), (9, 8)])

#hierarchical clustering using single-linkage
linked = linkage(X, method='single')

#dendrogram
plt.figure(figsize=(8, 4))
dendrogram(linked,
           orientation='top',
           distance_sort='ascending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


labels = fcluster(linked, t=2, criterion='maxclust')

#Plotting
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=100, edgecolors='k')
for i, point in enumerate(X):
    plt.text(point[0]+0.1, point[1], f"P{i+1}", fontsize=9)
plt.title("Clustered Data (2 Clusters)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
