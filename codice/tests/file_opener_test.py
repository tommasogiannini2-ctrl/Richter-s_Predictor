import unittest
from unittest.mock import patch
import pandas as pd
import os

# Sostituisci con l'import corretto del tuo modulo
from data_pipeline.file_opener import AbstractOpener, XLSOpener, CSVOpener, JSONOpener, scegli_opener


class DummyOpener(AbstractOpener):
    """Classe concreta fittizia usata esclusivamente per testare il metodo open() della classe base astratta."""

    def _load_data(self, path: str) -> pd.DataFrame:
        # Ritorna un piccolo DataFrame fittizio per confermare che il metodo è stato chiamato
        return pd.DataFrame({'test_col': [1, 2, 3]})


class TestDataOpeners(unittest.TestCase):
    """Suite di test unitari per le classi di lettura file e la Factory function."""

    @patch('os.path.exists')
    def test_abstract_opener_file_non_trovato(self, mock_exists):
        """Verifica che venga sollevata un'eccezione FileNotFoundError se il percorso non esiste fisicamente."""
        mock_exists.return_value = False  # Simuliamo che il file NON esista
        opener = DummyOpener()

        with self.assertRaises(FileNotFoundError) as context:
            opener.open("percorso/falso.csv")

        self.assertIn("non trovato", str(context.exception))

    @patch('os.path.exists')
    def test_abstract_opener_lettura_successo(self, mock_exists):
        """Verifica il corretto instradamento a _load_data se il file esiste nel filesystem."""
        mock_exists.return_value = True  # Simuliamo che il file ESISTA
        opener = DummyOpener()

        df = opener.open("percorso/vero.csv")

        # Verifichiamo che _load_data sia stato chiamato restituendo il df mockato
        self.assertEqual(list(df.columns), ['test_col'])
        self.assertEqual(len(df), 3)

    @patch('os.path.exists')
    def test_abstract_opener_cattura_eccezioni_pandas(self, mock_exists):
        """Verifica che eventuali errori di parsing interni (es. file corrotto) vengano incapsulati in un RuntimeError."""
        mock_exists.return_value = True
        opener = DummyOpener()

        # Forziamo intenzionalmente _load_data a generare un'eccezione generica
        with patch.object(opener, '_load_data', side_effect=ValueError("Formato csv corrotto")):
            with self.assertRaises(RuntimeError) as context:
                opener.open("file_corrotto.csv")

            self.assertIn("Errore durante la lettura del file", str(context.exception))

    # ==========================================
    # TEST DELLE CLASSI CONCRETE (Strategie)
    # ==========================================

    @patch('pandas.read_csv')
    def test_csv_opener_chiama_pandas(self, mock_read_csv):
        """Verifica che CSVOpener deleghi correttamente la lettura a pd.read_csv."""
        # Creiamo un finto DataFrame di ritorno
        mock_df = pd.DataFrame()
        mock_read_csv.return_value = mock_df

        opener = CSVOpener()
        risultato = opener._load_data("dati.csv")

        # Assicura che pandas sia stato chiamato col path corretto
        mock_read_csv.assert_called_once_with("dati.csv")
        self.assertIs(risultato, mock_df)

    @patch('pandas.read_excel')
    def test_xls_opener_chiama_pandas(self, mock_read_excel):
        """Verifica che XLSOpener deleghi correttamente la lettura a pd.read_excel."""
        opener = XLSOpener()
        opener._load_data("dati.xlsx")
        mock_read_excel.assert_called_once_with("dati.xlsx")

    @patch('pandas.read_json')
    def test_json_opener_chiama_pandas(self, mock_read_json):
        """Verifica che JSONOpener deleghi correttamente la lettura a pd.read_json."""
        opener = JSONOpener()
        opener._load_data("dati.json")
        mock_read_json.assert_called_once_with("dati.json")

    def test_scegli_opener_csv_e_txt(self):
        """Verifica che vengano assegnati i CSVOpener alle estensioni testuali."""
        self.assertIsInstance(scegli_opener("dataset.csv"), CSVOpener)
        self.assertIsInstance(scegli_opener("log_dati.txt"), CSVOpener)
        # Testa anche il case insensitive
        self.assertIsInstance(scegli_opener("FILE.CSV"), CSVOpener)

    def test_scegli_opener_excel(self):
        """Verifica l'istanziazione per fogli di calcolo Excel."""
        self.assertIsInstance(scegli_opener("tabella.xls"), XLSOpener)
        self.assertIsInstance(scegli_opener("nuova_tabella.xlsx"), XLSOpener)

    def test_scegli_opener_json(self):
        """Verifica l'istanziazione per file JSON."""
        self.assertIsInstance(scegli_opener("config.json"), JSONOpener)

    def test_scegli_opener_non_supportato(self):
        """Assicura che venga sollevata l'eccezione corretta per tipi di file non mappati."""
        with self.assertRaises(RuntimeError) as context:
            scegli_opener("immagine.png")

        self.assertIn("Tipo di file non supportato", str(context.exception))


if __name__ == '__main__':
    unittest.main()