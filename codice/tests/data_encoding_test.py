import unittest
from unittest.mock import patch
import pandas as pd

from data_pipeline.data_encoding import DataEncoding


class TestDataEncoding(unittest.TestCase):
    """Suite di test unitari per la classe DataEncoding, che gestisce la codifica One-Hot delle variabili categoriche."""

    def setUp(self):
        """Inizializza un DataFrame mock con feature miste (numeriche e categoriche) per simulare le operazioni di encoding e allineamento."""
        self.df_base = pd.DataFrame({
            'id': [1, 2, 3],
            'age': [10, 20, 30],
            'foundation_type': ['h', 'i', 'h'],
            'roof_type': ['n', 'q', 'x'],
            'damage_grade': [1, 2, 3]  # Variabile target
        })

    def test_init_copia_dataframe(self):
        """Garantisce che l'istanza della classe operi su una copia profonda dei dati, preservando l'integrità del DataFrame originale."""
        encoder = DataEncoding(self.df_base)
        self.assertIsNot(encoder.df, self.df_base)
        pd.testing.assert_frame_equal(encoder.df, self.df_base)

    @patch('builtins.print')
    def test_dummy_crea_colonne_one_hot(self, mock_print):
        """Verifica che le variabili categoriche dichiarate nel dominio vengano trasformate correttamente in dummy variables (0/1), rimuovendo le colonne originali."""
        encoder = DataEncoding(self.df_base)
        encoder.dummy()

        # Controlla che le colonne testuali originali siano state rimosse
        self.assertNotIn('foundation_type', encoder.df.columns)
        self.assertNotIn('roof_type', encoder.df.columns)

        # Controlla che le nuove colonne dummy siano presenti e denominate correttamente
        self.assertIn('foundation_type_h', encoder.df.columns)
        self.assertIn('foundation_type_i', encoder.df.columns)

        # Valida l'assegnazione accurata dei valori interi: la riga 0 aveva 'h', quindi deve avere 1.
        self.assertEqual(encoder.df.loc[0, 'foundation_type_h'], 1)
        self.assertEqual(encoder.df.loc[1, 'foundation_type_h'], 0)

        mock_print.assert_called()

    @patch('builtins.print')
    def test_trasforma_train_ignora_allineamento(self, mock_print):
        """Assicura che in fase di addestramento (is_train=True), la forzatura dell'allineamento delle feature venga by-passata."""
        encoder = DataEncoding(self.df_base, is_train=True)

        # Simuliamo una lista di colonne di riferimento che include una feature inesistente
        lista_train = ['id', 'age', 'foundation_type_h', 'colonna_fantasma']
        df_result = encoder.trasforma(lista_colonne=lista_train)

        # Poiché is_train=True, NON deve effettuare il reindex. Quindi 'colonna_fantasma' non deve essere aggiunta.
        self.assertNotIn('colonna_fantasma', df_result.columns)
        # La colonna 'roof_type_n' (generata da .dummy()) deve essere ancora presente, non essendo stata filtrata via.
        self.assertIn('roof_type_n', df_result.columns)

    @patch('builtins.print')
    def test_trasforma_test_esegue_allineamento(self, mock_print):
        """Verifica che in fase di inferenza (is_train=False), il set di test venga riallineato dinamicamente per corrispondere all'esatta topologia del train set."""
        encoder = DataEncoding(self.df_base, is_train=False)

        # Simula le colonne estratte dopo il fit sul set di addestramento.
        # Nota: 'foundation_type_w' NON è presente nel nostro df_base, 'roof_type_x' manca da questa lista.
        lista_colonne_train = [
            'id', 'age',
            'foundation_type_h', 'foundation_type_i', 'foundation_type_w',
            'damage_grade'
        ]

        df_result = encoder.trasforma(lista_colonne=lista_colonne_train)

        # Verifica 1: Le colonne assenti nel test ma presenti nel train devono essere create e riempite nativamente con 0.
        self.assertIn('foundation_type_w', df_result.columns)
        self.assertEqual(df_result['foundation_type_w'].sum(), 0)

        # Verifica 2: Le colonne categoriche presenti nel test ma non previste dal train devono essere omesse.
        self.assertNotIn('roof_type_n', df_result.columns)

        # Verifica 3: La colonna target ('damage_grade') deve essere esplicitamente esclusa dalla ricostruzione del test set come da logica del metodo.
        self.assertNotIn('damage_grade', df_result.columns)


if __name__ == '__main__':
    unittest.main()