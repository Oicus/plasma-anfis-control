import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans

def extract_rules(data_path, n_clusters=5):
    df = pd.read_parquet(data_path)
    X = df[["plasma_temp", "gamma_flux"]].values
    
    # Cluster centers using K-Means
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    centers = kmeans.cluster_centers_
    
    # Gaussian membership functions
    rules = []
    for i, center in enumerate(centers):
        rule = {
            "input1": ("plasma_temp", center[0], np.std(X[:,0])),
            "input2": ("gamma_flux", center[1], np.std(X[:,1])),
            "output": f"rule_{i}_output"
        }
        rules.append(rule)
    
    return rules

# Example usage
if __name__ == "__main__":
    rules = extract_rules("data/raw_sensor_data.parquet")
    print(f"Extracted {len(rules)} rules:")
    for i, rule in enumerate(rules):
        print(f"RULE {i}: IF {rule['input1'][0]}≈{rule['input1'][1]:.1f}±{rule['input1'][2]:.1f} AND ...")
