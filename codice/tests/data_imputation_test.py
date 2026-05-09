import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.impute import IterativeImputer, SimpleImputer

# Sostituisci con l'import corretto in base al nome del tuo file
from data_pipeline.data_imputation import DataImputation


class TestDataImputation(unittest.TestCase):
    """Suite di test unitari per la classe DataImputation, responsabile della gestione dei missing values."""

    def setUp(self):
        """Inizializza un DataFrame mock con colonne numeriche, categoriche e target per simulare l'imputazione."""
        self.df_base = pd.DataFrame({
            'building_id': [1, 2, 3, 4],
            'damage_grade': [1, np.nan, 3, 2],
            'area_percentage': [50, np.nan, 100, 75],  # Numerica con NaN
            'height_percentage': [10, 20, np.nan, 15],  # Numerica con NaN
            'roof_type': ['n', 'q', np.nan, 'n'],  # Categorica con NaN
            'foundation_type': ['h', 'h', 'i', 'h']  # Categorica senza NaN
        })

    def test_init_copia_dataframe(self):
        """Garantisce che l'istanza della classe operi su una copia profonda dei dati, preservando l'integrità del DataFrame originale."""
        imputer = DataImputation(self.df_base)
        self.assertIsNot(imputer.df, self.df_base)
        pd.testing.assert_frame_equal(imputer.df, self.df_base)

    @patch('builtins.print')
    def test_nessun_nan_salta_imputazione(self, mock_print):
        """Verifica l'early-exit: se il dataset non contiene alcun valore nullo, le operazioni di imputazione vengono bypassate."""
        df_pulito = pd.DataFrame({
            'area': [100, 200],
            'roof': ['n', 'q']
        })
        imputer = DataImputation(df_pulito)
        res_df = imputer.imputa()

        # Il dataframe deve rimanere invariato e gli imputer non devono essere inizializzati
        pd.testing.assert_frame_equal(res_df, df_pulito)
        self.assertIsNone(imputer.imputer_num)
        self.assertIsNone(imputer.imputer_cat)
        mock_print.assert_called()

    @patch('builtins.print')
    def test_colonne_escluse_ignorate(self, mock_print):
        """Verifica che le colonne 'building_id' e 'damage_grade' vengano esplicitamente escluse dall'imputazione numerica, conservando i loro eventuali NaN."""
        imputer = DataImputation(self.df_base, is_train=True)
        res_df = imputer.imputa()

        # damage_grade aveva un NaN all'indice 1. Deve essere ancora lì.
        self.assertTrue(pd.isna(res_df.loc[1, 'damage_grade']))

        # area_percentage aveva un NaN all'indice 1. Deve essere stato imputato.
        self.assertFalse(pd.isna(res_df.loc[1, 'area_percentage']))

    @patch('builtins.print')
    def test_imputazione_train_fit_transform(self, mock_print):
        """Verifica che, in modalità addestramento, vengano istanziati nuovi imputer e venga eseguito il fit_transform sui dati mancanti."""
        imputer = DataImputation(self.df_base, is_train=True)
        res_df = imputer.imputa()

        # Verifica che gli imputer siano stati creati e salvati nell'istanza
        self.assertIsInstance(imputer.imputer_num, IterativeImputer)
        self.assertIsInstance(imputer.imputer_cat, SimpleImputer)

        # Verifica l'avvenuta imputazione categorica (la moda di roof_type è 'n')
        self.assertEqual(res_df.loc[2, 'roof_type'], 'n')

        # Verifica che i NaN numerici siano scomparsi (escluso il target verificato nell'altro test)
        self.assertFalse(res_df['area_percentage'].isna().any())
        self.assertFalse(res_df['height_percentage'].isna().any())

    @patch('builtins.print')
    def test_imputazione_test_usa_solo_transform(self, mock_print):
        """Verifica che, in fase di inferenza (is_train=False), vengano usati gli imputer pre-addestrati e sia invocato solo il metodo transform."""

        # Creiamo dei Mock che simulano il comportamento degli imputer di sklearn
        mock_num = MagicMock()
        mock_cat = MagicMock()

        # Configuriamo i mock per restituire array numpy validi quando viene chiamato .transform()
        mock_num.transform.return_value = np.array([[60, 12], [80, 16]])
        mock_cat.transform.return_value = np.array([['q', 'i'], ['n', 'h']])

        # DataFrame di test piccolissimo
        df_test = pd.DataFrame({
            'building_id': [5, 6],
            'area_percentage': [np.nan, 80],
            'height_percentage': [12, np.nan],
            'roof_type': [np.nan, 'n'],
            'foundation_type': ['i', np.nan]
        })

        # Inizializziamo passando is_train=False e fornendo i nostri Mock
        imputer = DataImputation(df_test, imputer_num=mock_num, imputer_cat=mock_cat, is_train=False)
        imputer.imputa()

        # ASSERT PRINCIPALI:
        # Verifichiamo che il codice della classe abbia chiamato ESATTAMENTE .transform() sui nostri oggetti Mock
        mock_num.transform.assert_called_once()
        mock_cat.transform.assert_called_once()

        # Verifichiamo che .fit() o .fit_transform() NON siano mai stati chiamati
        mock_num.fit_transform.assert_not_called()
        mock_cat.fit_transform.assert_not_called()


if __name__ == '__main__':
    unittest.main()