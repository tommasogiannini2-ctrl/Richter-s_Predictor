import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self):
        self.model = None
        self.k_ottimale = None

    def plot_elbow_method(self, train_df, max_k=10, sample_size=30000, output_dir=None):
        """Trova il K ottimale usando solo i dati di Train."""
        sample = train_df.sample(n=min(len(train_df), sample_size), random_state=42)
        distortions = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
            km.fit(sample)
            distortions.append(km.inertia_)

        fig=plt.figure(figsize=(10, 5))
        plt.plot(K_range, distortions, 'bo-')
        plt.title('Metodo del Gomito (Dati di Train)')
        plt.tight_layout()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            percorso = os.path.join(output_dir, 'clustering_elbow.png')
            fig.savefig(percorso, dpi=150, bbox_inches='tight')
            print(f"    → Salvato: {percorso}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    def fit(self, train_df, k):
        """Addestra il modello e restituisce il One-Hot Encoding dei cluster."""
        self.k_ottimale = k
        self.model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)

        clusters = self.model.fit_predict(train_df)

        # Trasforma in One-Hot Encoding
        return pd.get_dummies(clusters, prefix='cluster', dtype=int)

    def predict(self, test_df):
        """Assegna i dati ai cluster esistenti e restituisce il One-Hot Encoding."""
        if self.model is None:
            raise Exception("Il modello non è stato addestrato. Esegui prima .fit()")

        clusters = self.model.predict(test_df)

        # Trasforma in One-Hot Encoding
        # Usiamo pd.get_dummies e poi ci assicuriamo che tutte le colonne esistano
        encoded = pd.get_dummies(clusters, prefix='cluster', dtype=int)

        # Se per caso un cluster non fosse presente nel test set,
        # aggiungiamo la colonna mancante con tutti zero per coerenza
        for i in range(self.k_ottimale):
            col_name = f'cluster_{i}'
            if col_name not in encoded.columns:
                encoded[col_name] = 0

        # Ordiniamo le colonne per sicurezza
        encoded = encoded.reindex(sorted(encoded.columns), axis=1)

        return encoded
