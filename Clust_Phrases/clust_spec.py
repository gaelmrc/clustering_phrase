import numpy as np
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel('simmat.xlsx', header = None, index_col = None)
matsim = df.to_numpy()
print('shape matsim =', np.shape(matsim))
#print(matsim[0])
matsim = np.array([matsim[k][0].split(',') for k in range (np.shape(matsim)[0])])
matsim = matsim.astype(float)
matsim = np.ones(np.shape(matsim)) - matsim
def spectral_clustering(matrix, k):

    start = time.time()
    spectral_clustering = SpectralClustering(n_clusters = k, affinity = "precomputed")
    labels = spectral_clustering.fit_predict(matrix)
    silhouette_avg = silhouette_score(matsim,labels)
    print("temps d'exécution =", time.time() - start)
    return labels, silhouette_avg
spectral_clustering(matsim, 2)

silhouettes = []
labels = []
for k in range(2,30):
    lab,sil = spectral_clustering(matsim,k)
    silhouettes.append(sil)
    labels.append(lab)

print("silhouettes = ",silhouettes)

sns.lineplot(silhouettes)
plt.xlabel('phrases')
plt.ylabel('silhouette score')
plt.savefig('silhouette_score.png')