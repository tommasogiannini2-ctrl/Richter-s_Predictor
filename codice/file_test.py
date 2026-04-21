import unittest
import pandas as pd
import numpy as np
from .data_reduction import DataReducer
from .data_pipeline.preprocessing import Preprocessing
from .data_pipeline.data_imputation import DataImputation


class TestDataReducer(unittest.TestCase):

    def setUp(self):
        """
        Questo metodo viene eseguito in automatico prima di ogni test.
        Creazione di un dataset fittizio con 10.000 record e uno sbilanciamento noto:
        - Grado 1: 10% (1.000 record)
        - Grado 2: 70% (7.000 record)
        - Grado 3: 20% (2.000 record)
        """
        print("\n[SetUp] Creazione del dataset stratificato...")
        dati = {
            'building_id': range(10000),
            'dummy_feature': [1] * 10000,
            'damage_grade': [1] * 1000 + [2] * 7000 + [3] * 2000
        }
        self.df_fake = pd.DataFrame(dati)

    def test_stratificazione_mantenuta(self):
        """
        Testa se la riduzione mantiene inalterate le proporzioni della colonna target.
        """
        print("\n[Test] Esecuzione verifica stratificazione...")

        # Calcolo le proporzioni originali (da 0.0 a 1.0)
        prop_originali = self.df_fake['damage_grade'].value_counts(normalize=True)

        # Istanzio il tuo riduttore e calcolo la memoria
        reducer = DataReducer(self.df_fake)
        n_record, memoria_attuale = reducer.get_info()

        # Imposto un limite di memoria molto basso per forzare un taglio drastico (taglio del 90%)
        limite_mb_test = memoria_attuale / 10

        # Eseguo la riduzione
        df_ridotto = reducer.riduci_per_memoria(limite_mb_test)

        # Calcolo le proporzioni dopo il taglio
        prop_ridotte = df_ridotto['damage_grade'].value_counts(normalize=True)

        # VERIFICA MATEMATICA
        for grado in [1, 2, 3]:
            # assertAlmostEqual verifica che i numeri siano uguali fino a un certo numero di decimali.
            # Se la stratificazione fallisce, il test si blocca qui e lancia un errore.
            self.assertAlmostEqual(
                prop_originali[grado],
                prop_ridotte[grado],
                places=2,
                msg=f"Fallimento sul Grado {grado}: Proporzione alterata!"
            )

        print(f"Test superato: Il dataset è passato da {len(self.df_fake)} a {len(df_ridotto)} record.")
        print("Le proporzioni delle classi sono rimaste identiche.")


class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        """
        Setup di un dataframe con valori nulli inseriti casualmente per testare la pulizia dei dati.
        """
        print("\n[SetUp] Creazione del dataset sporco...")
        np.random.seed(42) # Per riproducibilità
        n_rows = 1000
        
        # Creiamo un dataframe base con feature tipiche del problema
        data = {
            'building_id': range(n_rows),
            'age': np.random.randint(0, 100, n_rows),
            'area_percentage': np.random.randint(1, 100, n_rows),
            'height_percentage': np.random.randint(1, 30, n_rows),
            'foundation_type_r': np.random.choice([0, 1], n_rows),
            'foundation_type_w': np.random.choice([0, 1], n_rows),
            'damage_grade': np.random.choice([1, 2, 3], n_rows)
        }
        self.df_dirty = pd.DataFrame(data)
        
        # Inseriamo valori nulli a caso in alcune colonne (simulando circa il 10% di missing values)
        colonne_da_sporcare = ['age', 'area_percentage', 'height_percentage', 'foundation_type_r']
        
        for col in colonne_da_sporcare:
            # Scegliamo casualmente degli indici da impostare a NaN
            n_nulls = int(n_rows * 0.1)
            null_indices = np.random.choice(self.df_dirty.index, n_nulls, replace=False)
            self.df_dirty.loc[null_indices, col] = np.nan

    def test_presenza_valori_nulli(self):
        """
        Test di base per verificare che il dataframe contenga effettivamente valori nulli da pulire.
        """
        print("\n[Test] Verifica valori nulli nel dataset...")
        null_counts = self.df_dirty.isnull().sum()
        
        # Stampa le colonne che contengono nulli
        colonne_con_nulli = null_counts[null_counts > 0]
        print("Valori nulli per colonna:")
        print(colonne_con_nulli)
        
        # Assicuriamoci che ci siano valori nulli nel dataframe
        self.assertTrue(self.df_dirty.isnull().values.any(), "Il dataframe dovrebbe contenere valori nulli per il test.")
        print(f"Il dataframe contiene {self.df_dirty.isnull().sum().sum()} valori nulli totali.")

    def test_pulizia_nulli_rimozione_righe(self):
        """
        Testa la rimozione delle righe con valori nulli.
        """
        print("\n[Test] Pulizia valori nulli tramite rimozione righe...")

        # Copia il dataframe sporco per non alterare l'originale per altri test
        df = self.df_dirty.copy()
        df_rimozione = DataImputation(dataframe=df)
        df_rimozione.imputa()
        df_rimozione_processato = df_rimozione.df
        self.assertEqual(df_rimozione_processato.isnull().sum().sum(), 0, "Il dataframe pulito dovrebbe non contenere valori nulli (RIMOZIONE).")
        print(f"Il dataframe contiene {df_rimozione_processato.isnull().sum().sum()} valori nulli totali.")


if __name__ == '__main__':
    unittest.main()
