import numpy as np
import pandas as pd
import os
import logging
import json
from typing import List, Dict, Optional
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import validation_curve
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import dask.dataframe as dd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFuzzyRuleExtractor:
    def __init__(self, data_path: str, max_clusters: int = 10):
        """
        Geliştirilmiş bulanık kural çıkarıcı sınıfı
        
        Args:
            data_path (str): Veri dosya yolu (Parquet formatında)
            max_clusters (int): Maksimum küme sayısı
        """
        self.data_path = data_path
        self.max_clusters = max_clusters
        self.rules = []
        self._validate_inputs()
        self._load_data()
        self._normalize_data()
        
    def _validate_inputs(self):
        """Giriş parametrelerini doğrular"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.data_path}")
        if self.max_clusters < 1 or self.max_clusters > 20:
            raise ValueError("Geçersiz küme sayısı (1-20 arası olmalı)")

    def _load_data(self):
        """Veriyi Dask ile yükler ve ön işleme yapar"""
        try:
            self.ddf = dd.read_parquet(self.data_path)
            self.ddf = self.ddf[["plasma_temp", "gamma_flux"]].persist()
            self.X = self.ddf.compute().values
            logger.info(f"Veri başarıyla yüklendi. Örnek sayısı: {len(self.X)}")
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
            raise

    def _normalize_data(self):
        """Veriyi normalize eder"""
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.X_norm = (self.X - self.mean) / self.std
        logger.debug("Normalizasyon parametreleri:\nMean: %s\nStd: %s", self.mean, self.std)

    def _determine_optimal_clusters(self):
        """Optimal küme sayısını belirler"""
        cluster_range = np.arange(1, self.max_clusters+1)
        
        # Validation curve ile optimal küme sayısı
        train_scores, valid_scores = validation_curve(
            GaussianMixture(),
            self.X_norm,
            param_name="n_components",
            param_range=cluster_range,
            cv=3,
            n_jobs=-1
        )
        
        self.optimal_clusters = cluster_range[np.argmax(valid_scores.mean(axis=1))]
        logger.info(f"Optimal küme sayısı belirlendi: {self.optimal_clusters}")

    def _fit_gmm_model(self):
        """GMM modelini paralel şekilde eğitir"""
        self.gmm = GaussianMixture(
            n_components=self.optimal_clusters,
            n_init=15,
            covariance_type='full',
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        self.gmm.fit(self.X_norm)
        logger.info("GMM modeli başarıyla eğitildi")

    def _calculate_dynamic_thresholds(self):
        """Veriye dayalı dinamik eşikler belirler"""
        self.temp_quantiles = np.quantile(self.X[:,0], [0.25, 0.75])
        self.gamma_quantiles = np.quantile(self.X[:,1], [0.25, 0.75])

    def _calculate_consequent(self, cluster_mean: np.ndarray) -> str:
        """
        Dinamik eşiklere göre sonuç belirler
        
        Args:
            cluster_mean (np.ndarray): Küme ortalaması
            
        Returns:
            str: Durum etiketi
        """
        temp, gamma = cluster_mean
        
        if temp > self.temp_quantiles[1] and gamma < self.gamma_quantiles[0]:
            return "Optimal Fusion"
        elif temp > self.temp_quantiles[1] and gamma > self.gamma_quantiles[1]:
            return "Gamma Anomaly"
        elif temp < self.temp_quantiles[0]:
            return "Low Energy"
        else:
            return "Stable Operation"

    def extract_rules(self) -> List[Dict]:
        """Bulanık kuralları çıkarır"""
        self._determine_optimal_clusters()
        self._fit_gmm_model()
        self._calculate_dynamic_thresholds()
        
        means = self.gmm.means_ * self.std + self.mean
        covariances = self.gmm.covariances_ * np.outer(self.std, self.std)
        
        self.rules = []
        for i in range(self.optimal_clusters):
            rule = {
                'cluster_id': i+1,
                'parameters': {
                    'plasma_temp': {
                        'mean': float(means[i][0]),
                        'std': float(np.sqrt(covariances[i][0,0]))
                    },
                    'gamma_flux': {
                        'mean': float(means[i][1]),
                        'std': float(np.sqrt(covariances[i][1,1]))
                    },
                    'covariance_matrix': covariances[i].tolist(),
                    'weight': float(self.gmm.weights_[i])
                },
                'consequent': self._calculate_consequent(means[i])
            }
            self.rules.append(rule)
        
        logger.info(f"{len(self.rules)} adet kural başarıyla çıkarıldı")
        return self.rules

    def visualize_clusters(self, save_path: Optional[str] = None):
        """İnteraktif görselleştirme"""
        fig = px.scatter(
            x=self.X[:,0],
            y=self.X[:,1],
            color=self.gmm.predict(self.X_norm),
            labels={'x': 'Plasma Temperature (K)', 'y': 'Gamma Flux (MeV)'},
            title='Plasma State Clustering'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"İnteraktif görsel {save_path} kaydedildi")
        else:
            fig.show()

    def save_rules(self, file_path: str, indent: int = 2):
        """Kuralları JSON formatında kaydeder"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.rules, f, indent=indent)
            logger.info(f"Kurallar {file_path} dosyasına kaydedildi")
        except PermissionError:
            logger.error(f"Dosyaya yazma izni reddedildi: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Kayıt hatası: {str(e)}")
            raise

# Test senaryosu
if __name__ == "__main__":
    try:
        extractor = EnhancedFuzzyRuleExtractor(
            data_path="data/plasma_data.parquet",
            max_clusters=8
        )
        rules = extractor.extract_rules()
        extractor.visualize_clusters("interactive_clusters.html")
        extractor.save_rules("enhanced_rules.json")
        
        # Örnek kural gösterimi
        sample_rule = rules[0]
        print("\nÖrnek Kural:")
        print(f"Küme ID: {sample_rule['cluster_id']}")
        print(f"Sonuç: {sample_rule['consequent']}")
        print(f"Ağırlık: {sample_rule['parameters']['weight']:.3f}")
        
    except Exception as e:
        logger.error(f"Çalıştırma hatası: {str(e)}")
