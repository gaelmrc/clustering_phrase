import numpy as np
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
matsim = 0.5 * (matsim + matsim.T)

def spectral_clustering(matrix, k):

    start = time.time()
    spectral_clustering = SpectralClustering(n_clusters = k, affinity = "precomputed")
    labels = spectral_clustering.fit_predict(matrix)
    print( "shape label = ", np.shape(labels))
    silhouette_avg = silhouette_score(matrix,labels)
    db_score = davies_bouldin_score(matrix, labels)

    print("temps d'exécution =", time.time() - start)
    return labels, silhouette_avg, db_score
spectral_clustering(matsim, 2)

silhouettes = []
db_score = []
labels = []
for k in range(2,100):
    lab,sil,db = spectral_clustering(matsim,k)
    silhouettes.append(sil)
    labels.append(lab)
    db_score.append(db)

print("silhouettes = ",silhouettes)
sns.set_theme()
sns.lineplot(x = range(len(silhouettes)),y = silhouettes, color = "orange")
plt.xlabel('nombre de clusters')
plt.legend(labels = ["silhouette score"])
plt.ylabel('silhouette score')
plt.savefig('silhouette_score.png')
plt.clf()

sns.lineplot(x = range(len(db_score)), y = db_score, color = "blue")
plt.xlabel('nombre de clusters')
plt.legend(labels = ["Davies-Bouldin score"])
plt.ylabel('Davies-Bouldin score')
plt.savefig('Davies_Bouldin_score.png')