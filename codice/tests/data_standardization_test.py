import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from data_pipeline.data_standardization import COLONNE_CONTINUE, DataScaling


class TestDataScaling(unittest.TestCase):
    """Suite di test unitari per la classe DataScaling, che gestisce la standardizzazione delle feature continue."""

    def setUp(self):
        """Prepara un DataFrame misto per verificare che la standardizzazione colpisca solo le colonne designate."""
        self.df_base = pd.DataFrame({
            'building_id': [1, 2, 3],  # Da ignorare
            'age': [10, 50, 30],  # Continua target
            'area_percentage': [20, 80, 50],  # Continua target
            'foundation_type_h': [1, 0, 1],  # Dummy (da ignorare)
            'damage_grade': [1, 3, 2]  # Target (da ignorare)
        })

    def test_init_copia_dataframe(self):
        """Garantisce che la classe operi su una copia dei dati e inizializzi correttamente lo StandardScaler di default se non fornito."""
        scaler_obj = DataScaling(self.df_base)
        self.assertIsNot(scaler_obj.df, self.df_base)
        pd.testing.assert_frame_equal(scaler_obj.df, self.df_base)
        self.assertIsInstance(scaler_obj.scaler, StandardScaler)

    @patch('builtins.print')
    def test_nessuna_colonna_salta_standardizzazione(self, mock_print):
        """Verifica che se il DataFrame non contiene feature continue previste, il processo termini restituendo il df intatto senza alterare nulla."""
        df_vuoto = pd.DataFrame({
            'building_id': [1, 2],
            'categorica': ['a', 'b']
        })
        scaler_obj = DataScaling(df_vuoto)
        res_df = scaler_obj.standardizza()

        # Nessuna trasformazione, lo scaler non deve essere chiamato
        pd.testing.assert_frame_equal(res_df, df_vuoto)
        mock_print.assert_called()

    @patch('builtins.print')
    def test_standardizza_train_esegue_fit_transform(self, mock_print):
        """Verifica che in fase di training venga invocato fit_transform esclusivamente sulle feature continue presenti."""
        mock_scaler = MagicMock()
        # Simuliamo il risultato della standardizzazione con una matrice numpy
        mock_scaler.fit_transform.return_value = np.array([[-1, -1], [1, 1], [0, 0]])

        scaler_obj = DataScaling(self.df_base, scaler=mock_scaler, is_train=True)
        res_df = scaler_obj.standardizza()

        # Controlliamo che il fit_transform sia stato chiamato sui dati corretti (solo age e area_percentage)
        mock_scaler.fit_transform.assert_called_once()

        # Le colonne da ignorare devono essere rimaste intatte
        self.assertEqual(list(res_df['building_id']), [1, 2, 3])
        self.assertEqual(list(res_df['foundation_type_h']), [1, 0, 1])

        # Le colonne standardizzate devono contenere i nuovi valori dell'array mockato
        self.assertEqual(list(res_df['age']), [-1, 1, 0])
        self.assertEqual(list(res_df['area_percentage']), [-1, 1, 0])

    def test_colonne_continue_condivise_per_scaling_e_clustering(self):
        """Verifica l'elenco condiviso delle feature continue usato anche dal K-Means."""
        self.assertEqual(
            COLONNE_CONTINUE,
            [
                'age',
                'area_percentage',
                'height_percentage',
                'count_floors_pre_eq',
                'count_families'
            ]
        )

    @patch('builtins.print')
    def test_standardizza_test_esegue_solo_transform(self, mock_print):
        """Verifica che in fase di validazione/test venga usato lo scaler addestrato chiamando unicamente transform."""
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0, 0], [0.5, 0.5], [-0.5, -0.5]])

        scaler_obj = DataScaling(self.df_base, scaler=mock_scaler, is_train=False)
        scaler_obj.standardizza()

        # Verifica la correttezza delle chiamate (solo transform, mai fit)
        mock_scaler.transform.assert_called_once()
        mock_scaler.fit_transform.assert_not_called()
        mock_scaler.fit.assert_not_called()

    def test_standardizza_test_senza_scaler_solleva_errore(self):
        """Assicura che la mancata iniezione di uno scaler pre-addestrato per il set di test generi un ValueError esplicito."""
        # Creiamo l'oggetto is_train=False ma scaler=None
        scaler_obj = DataScaling(self.df_base, scaler=None, is_train=False)

        # Verifichiamo che chiamando standardizza venga sollevata l'eccezione
        with self.assertRaises(ValueError) as context:
            scaler_obj.standardizza()

        self.assertEqual(str(context.exception), "Sul test è obbligatorio passare lo scaler addestrato sul train.")


if __name__ == '__main__':
    unittest.main()
