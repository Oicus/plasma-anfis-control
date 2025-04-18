import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

def extract_fuzzy_rules(data_path, max_clusters=8):
    # Veri yükleme ve ön işleme
    df = pd.read_parquet(data_path)
    X = df[["plasma_temp", "gamma_flux"]].values
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalizasyon
    
    # Optimal küme sayısı belirleme
    bic_scores = []
    for n in range(1, max_clusters+1):
        gmm = GaussianMixture(n_components=n)
        gmm.fit(X_normalized)
        bic_scores.append(gmm.bic(X_normalized))
    
    optimal_clusters = np.argmin(bic_scores) + 1
    
    # GMM ile kümeleme
    gmm = GaussianMixture(n_components=optimal_clusters)
    gmm.fit(X_normalized)
    
    # Üyelik fonksiyonları ve kurallar
    rules = []
    for i in range(optimal_clusters):
        mean = gmm.means_[i] * X.std(axis=0) + X.mean(axis=0)  # Denormalize
        cov = gmm.covariances_[i] * np.outer(X.std(axis=0), X.std(axis=0))
        
        rule = {
            'antecedent': {
                'plasma_temp': ('gaussian', mean[0], np.sqrt(cov[0,0])),
                'gamma_flux': ('gaussian', mean[1], np.sqrt(cov[1,1])),
                'cov_matrix': cov  # İlişki için
            },
            'consequent': self._calculate_consequent(mean),
            'membership': gmm.weights_[i]
        }
        rules.append(rule)
    
    return rules

def _calculate_consequent(self, mean):
    # Örnek: Sıcaklık ve Gamma arasındaki ilişkiye göre çıktı üret
    if mean[0] > 5000 and mean[1] < 100:
        return "plasma_stable"
    elif mean[1] > 200:
        return "gamma_warning"
    else:
        return "normal_operation"

# Görselleştirme
def plot_cluster_distributions(rules):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    
    # Sıcaklık dağılımları
    for i, rule in enumerate(rules):
        mu = rule['antecedent']['plasma_temp'][1]
        sigma = rule['antecedent']['plasma_temp'][2]
        x = np.linspace(mu-3*sigma, mu+3*sigma, 100)
        ax[0].plot(x, stats.norm.pdf(x, mu, sigma), 
                 label=f'Rule {i}')
    
    # Gamma-Isı korelasyonu
    for rule in rules:
        cov = rule['antecedent']['cov_matrix']
        plt.scatter(cov[0,1], rule['membership'], 
                  s=100*rule['membership'])
    
    plt.show()
