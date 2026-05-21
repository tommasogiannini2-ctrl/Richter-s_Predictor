import unittest
import sys
import os
import pandas as pd
import numpy as np
import shutil
import tempfile
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Aggiungi la directory principale al path per gli import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_evaluation.train_model import _build_model, run as run_training
from model_evaluation.validation import _build_search_space, FeatureSelectionSearch

class TestAdaBoostIntegration(unittest.TestCase):
    """
    Test suite per verificare l'integrazione dell'algoritmo AdaBoost nel progetto.
    """

    def setUp(self):
        """Crea una directory temporanea per i file di output del test."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Rimuove la directory temporanea dopo il test."""
        shutil.rmtree(self.test_dir)

    def test_build_model_ada(self):
        """Verifica che _build_model istanzi correttamente AdaBoostClassifier."""
        model = _build_model(
            model="ada",
            n_estimators=100,
            learning_rate=0.1,
            base_estimator_max_depth=2
        )
        
        self.assertIsInstance(model, AdaBoostClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertIsInstance(model.estimator, DecisionTreeClassifier)
        self.assertEqual(model.estimator.max_depth, 2)

    def test_search_space_contains_ada(self):
        """Verifica che lo spazio di ricerca contenga configurazioni per AdaBoost."""
        spazio = _build_search_space(include_sfs=False)
        
        # Cerchiamo quanti dizionari usano AdaBoostClassifier
        ada_configs = [d for d in spazio if d["model"][0].__class__ == AdaBoostClassifier]
        
        # Dovrebbero esserci 4 configurazioni (una per ogni selettore non-SFS)
        self.assertEqual(len(ada_configs), 4)
        
        # Verifica che i parametri siano quelli attesi
        for config in ada_configs:
            self.assertIn("model__n_estimators", config)
            self.assertIn("model__learning_rate", config)
            self.assertIn("model__estimator", config)

    def test_full_training_run_ada(self):
        """
        Verifica che la funzione run di train_model funzioni end-to-end con AdaBoost.
        Simula il caricamento dei file finali e l'addestramento.
        """
        # 1. Creazione dati fittizi "finali" (quelli che main.py salverebbe in output)
        df_train = pd.DataFrame(np.random.rand(50, 10), columns=[f"f{i}" for i in range(10)])
        df_train["damage_grade"] = np.random.randint(1, 4, 50)
        
        df_val = pd.DataFrame(np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)])
        df_val["damage_grade"] = np.random.randint(1, 4, 20)
        
        df_test = pd.DataFrame(np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)])
        df_test["damage_grade"] = np.random.randint(1, 4, 20)

        # Dataset per submission (senza target, ma con building_id)
        df_uff = pd.DataFrame(np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)])
        df_uff.insert(0, "building_id", range(1000, 1020))

        # Salvataggio CSV nella cartella temporanea
        df_train.to_csv(os.path.join(self.test_dir, "train_finale.csv"), index=False)
        df_val.to_csv(os.path.join(self.test_dir, "val_finale.csv"), index=False)
        df_test.to_csv(os.path.join(self.test_dir, "test_finale.csv"), index=False)
        df_uff.to_csv(os.path.join(self.test_dir, "test_ufficiale_processato.csv"), index=False)

        # 2. Esecuzione della funzione run (il cuore del training finale)
        # Usiamo parametri ridotti per velocità
        try:
            run_training(
                model="ada",
                output_dir=self.test_dir,
                n_estimators=10,
                learning_rate=1.0,
                base_estimator_max_depth=1,
                no_proba=True  # Velocizziamo evitando le curve ROC
            )
        except Exception as e:
            self.fail(f"train_model.run ha sollevato un'eccezione con AdaBoost: {e}")

        # 3. Verifiche
        # Controllo che il modello sia stato salvato
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "model_finale.pkl")))
        # Controllo che la submission sia stata generata
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "submission.csv")))
        # Controllo che la cartella eval/validation esista
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "eval", "validation")))

    def test_feature_selection_search_with_ada(self):
        """Test di integrazione: verifica che FeatureSelectionSearch funzioni con AdaBoost."""
        from sklearn.datasets import make_classification
        
        # Dataset sintetico piccolo
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_classes=3,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
        y_s = pd.Series(y + 1)
        
        # Ricerca limitata
        search = FeatureSelectionSearch(
            n_iter=5,
            cv=2,
            include_sfs=False,
            random_state=42,
            verbose=0
        )
        
        # Eseguiamo il fit
        try:
            search.fit(X_df, y_s)
        except Exception as e:
            self.fail(f"FeatureSelectionSearch.fit ha sollevato un'eccezione: {e}")
            
        # Verifica che i risultati contengano AdaBoostClassifier se è stato campionato
        results = search.get_results()
        self.assertTrue("AdaBoostClassifier" in results["model"].values or len(results) >= 1)

if __name__ == "__main__":
    unittest.main()

