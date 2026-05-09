import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Serve per usare l'imputer avanzato di scikit-learn senza errori
from sklearn.experimental import enable_iterative_imputer

# Importazione corretta della classe dal tuo pacchetto
from data_pipeline.preprocessing import Preprocessing, dividi_train_validation_test


class TestPreprocessing(unittest.TestCase):
    """Test per la classe Preprocessing.
    Verifica che le 4 fasi (pulizia, imputazione, codifica, standardizzazione)
    vengano eseguite nel giusto ordine e senza errori.
    """

    def setUp(self):
        """Crea un piccolo DataFrame di base da usare in tutti i test."""
        self.df_base = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'damage_grade': [1, 2, 3]
        })

    def test_init_imposta_default(self):
        """Verifica che la classe parta con i valori iniziali giusti (es. liste vuote)."""
        orchestrator = Preprocessing(self.df_base)

        self.assertTrue(orchestrator.is_train)
        self.assertIsNotNone(orchestrator.scaler)
        self.assertEqual(orchestrator.colonne_eliminate, [])
        self.assertEqual(orchestrator.lista_colonne, [])
        self.assertIsNone(orchestrator.imputer_num)

    # ==========================================
    # TEST DEL FLUSSO TRAIN
    # ==========================================

    @patch('builtins.print')
    @patch('data_pipeline.preprocessing.DataScaling')
    @patch('data_pipeline.preprocessing.DataEncoding')
    @patch('data_pipeline.preprocessing.DataImputation')
    @patch('data_pipeline.preprocessing.DataCleaning')
    def test_esegui_train_flusso_completo(self, MockCleaning, MockImputation, MockEncoding, MockScaling, mock_print):
        """Verifica il flusso per i dati di Addestramento (Train).
        Controlla che vengano salvate le informazioni importanti (es. lo scaler usato
        o le colonne cancellate) per poterle riusare dopo sul test.
        """
        # Creiamo dei finti risultati per le classi interne
        istanza_cleaning = MockCleaning.return_value
        istanza_cleaning.pulisci.return_value = self.df_base.copy()
        istanza_cleaning.colonne_eliminate = ['col_brutta']

        istanza_imputation = MockImputation.return_value
        istanza_imputation.imputa.return_value = self.df_base.copy()
        istanza_imputation.imputer_num = "MockImputerNum"
        istanza_imputation.imputer_cat = "MockImputerCat"

        istanza_encoding = MockEncoding.return_value
        istanza_encoding.trasforma.return_value = self.df_base.copy()

        istanza_scaling = MockScaling.return_value
        istanza_scaling.standardizza.return_value = self.df_base.copy()
        istanza_scaling.scaler = "MockScaler"

        # Facciamo partire il processo
        orchestrator = Preprocessing(self.df_base, is_train=True)
        orchestrator.esegui()

        # Controlliamo che abbia chiamato le funzioni giuste
        istanza_cleaning.elimina_record_null_percentuale.assert_called_once()
        istanza_cleaning.elimina_colonne_nulle.assert_called_once()

        # Controlliamo che abbia salvato le regole calcolate
        self.assertEqual(orchestrator.colonne_eliminate, ['col_brutta'])
        self.assertEqual(orchestrator.imputer_num, "MockImputerNum")
        self.assertEqual(orchestrator.imputer_cat, "MockImputerCat")
        self.assertEqual(orchestrator.scaler, "MockScaler")
        self.assertEqual(orchestrator.lista_colonne, list(self.df_base.columns))

    # ==========================================
    # TEST DEL FLUSSO TEST
    # ==========================================

    @patch('builtins.print')
    @patch('data_pipeline.preprocessing.DataScaling')
    @patch('data_pipeline.preprocessing.DataEncoding')
    @patch('data_pipeline.preprocessing.DataImputation')
    @patch('data_pipeline.preprocessing.DataCleaning')
    def test_esegui_test_flusso_completo(self, MockCleaning, MockImputation, MockEncoding, MockScaling, mock_print):
        """Verifica il flusso per i dati di Test.
        Controlla che riapplichi esattamente le stesse modifiche fatte nel Train,
        senza ricalcolare nulla da zero.
        """
        istanza_cleaning = MockCleaning.return_value
        istanza_cleaning.pulisci.return_value = self.df_base.copy()

        istanza_imputation = MockImputation.return_value
        istanza_imputation.imputa.return_value = self.df_base.copy()

        istanza_encoding = MockEncoding.return_value
        istanza_encoding.trasforma.return_value = self.df_base.copy()

        istanza_scaling = MockScaling.return_value
        istanza_scaling.standardizza.return_value = self.df_base.copy()

        # Simuliamo di avere già le regole salvate dal Train
        orchestrator = Preprocessing(
            self.df_base,
            is_train=False,
            colonne_eliminate=['col_rimossa_prima'],
            lista_colonne=['col1', 'col2', 'col3']
        )
        orchestrator.esegui()

        # Deve eliminare le colonne che avevamo già deciso di togliere nel Train
        istanza_cleaning.applica_colonne_eliminate.assert_called_once_with(['col_rimossa_prima'])
        # Non deve mai calcolare regole nuove
        istanza_cleaning.elimina_colonne_nulle.assert_not_called()

    # ==========================================
    # TEST GESTIONE TARGET
    # ==========================================

    @patch('builtins.print')
    @patch('data_pipeline.preprocessing.DataScaling')
    @patch('data_pipeline.preprocessing.DataEncoding')
    @patch('data_pipeline.preprocessing.DataImputation')
    @patch('data_pipeline.preprocessing.DataCleaning')
    def test_salvataggio_target_grade(self, MockCleaning, MockImputation, MockEncoding, MockScaling, mock_print):
        """Controlla che la colonna target ('damage_grade') non venga rovinata o modificata per errore durante la pulizia."""

        # Simuliamo un errore in una delle fasi che rovina la colonna target
        df_rotto = self.df_base.copy()
        df_rotto['damage_grade'] = [99, 99, 99]
        MockScaling.return_value.standardizza.return_value = df_rotto

        MockCleaning.return_value.pulisci.return_value = self.df_base.copy()
        MockImputation.return_value.imputa.return_value = self.df_base.copy()
        MockEncoding.return_value.trasforma.return_value = self.df_base.copy()

        orchestrator = Preprocessing(self.df_base)
        risultato = orchestrator.esegui()

        # Alla fine, la colonna target deve essere rimessa a posto coi valori originali (1, 2, 3)
        self.assertEqual(list(risultato['damage_grade']), [1, 2, 3])


class TestDividiTrainValidationTest(unittest.TestCase):
    """Test per la funzione che divide il dataset in tre parti."""

    @patch('builtins.print')
    def test_split_proporzioni_corrette(self, mock_print):
        """Verifica che il dataset venga diviso esattamente nel rapporto 70% (Train), 15% (Val) e 15% (Test) senza perdere righe."""
        # Creiamo 100 righe fittizie per fare i calcoli percentuali esatti
        df_100 = pd.DataFrame({
            'feature': range(100),
            'damage_grade': [1, 2, 3, 1] * 25
        })

        train, val, test = dividi_train_validation_test(df_100, target_column='damage_grade')

        # Controlliamo che la somma faccia sempre 100 (nessuna riga persa)
        self.assertEqual(len(train) + len(val) + len(test), 100)

        # Controlliamo le percentuali esatte
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)


if __name__ == '__main__':
    unittest.main()