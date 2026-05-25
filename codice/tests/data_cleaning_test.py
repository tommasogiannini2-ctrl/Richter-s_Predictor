import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from data_pipeline.data_cleaning import DataCleaning


class TestDataCleaning(unittest.TestCase):
    """Suite di test unitari per la verifica della logica di pulizia del dataset."""

    def setUp(self):
        """Inizializza un DataFrame mock rappresentativo per testare i vari scenari di pulizia (valori anomali, nulli, tipi invalidi)."""
        self.df_base = pd.DataFrame({
            'age': [10, 850, -5, 45],
            'count_floors_pre_eq': [2, 16, 0, 3],
            'area_percentage': [50, 150, 0, 20],
            'count_families': [1, -1, 2, 0],
            'damage_grade': [1, np.nan, 3, 2],
            'foundation_type': ['h', 'z', 'r', 'u'],
            'has_superstructure_adobe_mud': [1, 0, 2, 1]
        })

    def test_init_copia_dataframe(self):
        """Garantisce che l'istanza della classe operi su una copia profonda dei dati, preservando il DataFrame originale."""
        cleaner = DataCleaning(self.df_base)
        self.assertIsNot(cleaner.df, self.df_base)
        pd.testing.assert_frame_equal(cleaner.df, self.df_base)

    @patch('builtins.print')
    def test_elimina_duplicati(self, mock_print):
        """Verifica l'identificazione e la rimozione accurata delle righe esattamente duplicate nel dataset."""
        df_con_duplicati = pd.DataFrame({
            'id': [1, 1, 2, 3, 3],
            'val': ['a', 'a', 'b', 'c', 'c']
        })
        cleaner = DataCleaning(df_con_duplicati)
        cleaner.elimina_duplicati()

        self.assertEqual(len(cleaner.df), 3)
        mock_print.assert_called()

    @patch('builtins.print')
    def test_pulisci_variabili(self, mock_print):
        """Verifica l'applicazione dei vincoli di dominio logico sulle variabili numeriche, sostituendo i valori fuori range con pd.NA."""
        cleaner = DataCleaning(self.df_base)
        cleaner.pulisci_variabili()

        self.assertTrue(pd.isna(cleaner.df.loc[1, 'age']))
        self.assertTrue(pd.isna(cleaner.df.loc[2, 'age']))
        self.assertEqual(cleaner.df.loc[0, 'age'], 10)
        self.assertTrue(pd.isna(cleaner.df.loc[1, 'count_floors_pre_eq']))
        self.assertTrue(pd.isna(cleaner.df.loc[2, 'count_floors_pre_eq']))



    @patch('builtins.print')
    def test_rimuovi_outlier_strutturali_train(self, mock_print):
        """Assicura che, in fase di training, le righe contenenti valori categorici non previsti dal dominio vengano scartate integralmente."""
        cleaner = DataCleaning(self.df_base, is_train=True)
        cleaner.rimuovi_outlier_strutturali()

        self.assertEqual(len(cleaner.df), 2)
        self.assertTrue(all(cleaner.df['foundation_type'].isin(['h', 'r', 'u'])))

    @patch('builtins.print')
    def test_rimuovi_outlier_strutturali_test(self, mock_print):
        """Assicura che, in fase di inferenza (test), le anomalie strutturali vengano imputate come pd.NA senza alterare la cardinalità delle righe."""
        cleaner = DataCleaning(self.df_base, is_train=False)
        cleaner.rimuovi_outlier_strutturali()

        self.assertEqual(len(cleaner.df), 4)
        self.assertTrue(pd.isna(cleaner.df.loc[1, 'foundation_type']))
        self.assertTrue(pd.isna(cleaner.df.loc[2, 'has_superstructure_adobe_mud']))

    @patch('builtins.print')
    def test_elimina_record_null_percentuale(self, mock_print):
        """Verifica l'eliminazione dei record che superano la soglia massima consentita (es. 50%) di valori mancanti per riga."""
        df = pd.DataFrame({
            'a': [1, np.nan, np.nan],
            'b': [2, np.nan, np.nan],
            'c': [3, np.nan, 3],
            'd': [4, np.nan, np.nan]
        })
        cleaner = DataCleaning(df)
        cleaner.elimina_record_null_percentuale(soglia_percentuale=0.50)

        self.assertEqual(len(cleaner.df), 1)

    @patch('builtins.print')
    def test_elimina_colonne_nulle(self, mock_print):
        """Valida il drop delle intere feature (colonne) la cui densità di nulli è superiore alla tolleranza specificata."""
        df = pd.DataFrame({
            'buona': [1, 2, 3, 4],
            'pessima': [1, np.nan, np.nan, np.nan]
        })
        cleaner = DataCleaning(df)
        cleaner.elimina_colonne_nulle(soglia_percentuale=0.5)

        self.assertIn('buona', cleaner.df.columns)
        self.assertNotIn('pessima', cleaner.df.columns)
        self.assertEqual(cleaner.colonne_eliminate, ['pessima'])

    def test_pulisci_chiama_metodi_sequenza_train(self):
        """Verifica tramite patch degli oggetti che il flusso principale ('pulisci') esegua tutte le pipeline previste per il set di training."""
        cleaner = DataCleaning(self.df_base, is_train=True)

        with patch.object(cleaner, 'elimina_duplicati') as mock_dup, \
                patch.object(cleaner, 'pulisci_variabili') as mock_var, \
                patch.object(cleaner, 'elimina_classnull') as mock_classnull, \
                patch.object(cleaner, 'rimuovi_outlier_strutturali') as mock_out:
            cleaner.pulisci()

            mock_dup.assert_called_once()
            mock_var.assert_called_once()
            mock_classnull.assert_called_once()
            mock_out.assert_called_once()

    def test_pulisci_chiama_metodi_sequenza_test(self):
        """Verifica che il metodo 'elimina_classnull' sia correttamente disabilitato nel flusso principale quando si elabora il set di test."""
        cleaner = DataCleaning(self.df_base, is_train=False)

        with patch.object(cleaner, 'elimina_duplicati'), \
                patch.object(cleaner, 'pulisci_variabili'), \
                patch.object(cleaner, 'elimina_classnull') as mock_classnull, \
                patch.object(cleaner, 'rimuovi_outlier_strutturali'):
            cleaner.pulisci()

            mock_classnull.assert_not_called()


if __name__ == '__main__':
    unittest.main()