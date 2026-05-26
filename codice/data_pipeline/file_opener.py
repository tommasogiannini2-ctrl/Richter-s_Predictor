import pandas as pd
import os
from abc import ABC, abstractmethod


class AbstractOpener(ABC):
    """
    Classe astratta che funge da interfaccia per i lettori di file.
    Garantisce la corretta validazione dell'esistenza del file e gestisce
    le eccezioni risalenti a pandas in modo centralizzato.
    """

    def open(self, dataframe_path: str) -> pd.DataFrame:
        """
        Controlla l'esistenza del file e ne coordina il caricamento sicuro.

        Input:
          - dataframe_path (str): Percorso del file da caricare.

        Output:
          - pd.DataFrame: DataFrame caricato in memoria.
        """
        if not os.path.exists(dataframe_path):
            raise FileNotFoundError(f"File {dataframe_path} non trovato")
        try:
            return self._load_data(dataframe_path)
        except Exception as e:
            raise RuntimeError(f"Errore durante la lettura del file {dataframe_path}: {e}")

    @abstractmethod
    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Metodo protetto astratto implementato dalle classi derivate specifiche.
        """
        pass


class XLSOpener(AbstractOpener):
    """
    Lettore specifico per fogli di calcolo in formato Excel (.xls, .xlsx).
    """
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_excel(path)


class CSVOpener(AbstractOpener):
    """
    Lettore specifico per file in formato CSV (.csv, .txt).
    """
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)


class JSONOpener(AbstractOpener):
    """
    Lettore specifico per file in formato JSON (.json).
    """
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_json(path)


def scegli_opener(dataframe_path: str) -> AbstractOpener:
    """
    Factory function per selezionare dinamicamente il lettore corretto in base all'estensione del file.

    Input:
      - dataframe_path (str): Percorso o nome del file da cui estrarre l'estensione.

    Output:
      - AbstractOpener: Oggetto lettore specializzato corrispondente all'estensione.
    """
    ext = dataframe_path.split('.')[-1].lower()
    match ext:
        case 'csv' | 'txt':
            return CSVOpener()
        case 'xls' | 'xlsx':
            return XLSOpener()
        case 'json':
            return JSONOpener()
        case _:
            raise RuntimeError(f"Tipo di file non supportato: {ext}")
