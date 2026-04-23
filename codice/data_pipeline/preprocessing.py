import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_cleaning import DataCleaning
from data_imputation import DataImputation
from data_encoding import DataEncoding
from data_standardization import DataScaling


class Preprocessing:
    """
    Orchestratore principale della pipeline di preprocessing.
    Coordina in sequenza: pulizia, imputazione, encoding e standardizzazione.
    """

    def __init__(self, dataframe: pd.DataFrame, scaler=None, imputer_num=None,
                 imputer_cat=None, colonne_eliminate=None, lista_colonne=None, is_train=True):
        self.df = dataframe.copy()
        self.is_train = is_train

        self.scaler = scaler if scaler is not None else StandardScaler()
        self.imputer_num = imputer_num
        self.imputer_cat = imputer_cat
        self.colonne_eliminate = colonne_eliminate if colonne_eliminate is not None else []
        self.lista_colonne = lista_colonne if lista_colonne is not None else []

    def esegui(self) -> pd.DataFrame:
        """Esegue l'intera pipeline di preprocessing."""
        modalita = 'TRAIN' if self.is_train else 'TEST'

        print(f"\n{'=' * 60}")
        print(f"  PREPROCESSING — {modalita}")
        print(f"{'=' * 60}")

        # ── FASE 1: PULIZIA ──────────────────────────────────────────
        print(f"\n  [1/4] Pulizia dei dati")
        print(f"  {'-' * 48}")
        cleaning = DataCleaning(self.df, is_train=self.is_train)

        if self.is_train:
            cleaning.elimina_record_null_percentuale()
            cleaning.elimina_colonne_nulle()
            self.colonne_eliminate = cleaning.colonne_eliminate
        else:
            if self.colonne_eliminate:
                cleaning.applica_colonne_eliminate(self.colonne_eliminate)

        self.df = cleaning.pulisci()

        # ── FASE 2: IMPUTAZIONE ──────────────────────────────────────
        print(f"\n  [2/4] Imputazione valori mancanti")
        print(f"  {'-' * 48}")
        imputation = DataImputation(
            self.df,
            imputer_num=self.imputer_num,
            imputer_cat=self.imputer_cat,
            is_train=self.is_train
        )
        self.df = imputation.imputa()

        if self.is_train:
            self.imputer_num = imputation.imputer_num
            self.imputer_cat = imputation.imputer_cat

        # ── FASE 3: ENCODING ─────────────────────────────────────────
        print(f"\n  [3/4] Encoding variabili categoriche")
        print(f"  {'-' * 48}")
        encoding = DataEncoding(self.df, is_train=self.is_train)
        self.df = encoding.trasforma(lista_colonne=self.lista_colonne)

        # ── FASE 4: STANDARDIZZAZIONE ────────────────────────────────
        print(f"\n  [4/4] Standardizzazione variabili continue")
        print(f"  {'-' * 48}")
        scaling = DataScaling(self.df, scaler=self.scaler, is_train=self.is_train)
        self.df = scaling.standardizza()

        if self.is_train:
            self.scaler = scaling.scaler
            self.lista_colonne = self.df.columns.tolist()

        # ── RIEPILOGO FINALE ─────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  Preprocessing {modalita} completato")
        print(f"  {'Righe:':<40} {self.df.shape[0]:>8}")
        print(f"  {'Colonne:':<40} {self.df.shape[1]:>8}")
        print(f"  {'Valori mancanti residui:':<40} {self.df.isnull().sum().sum():>8}")
        print(f"{'=' * 60}\n")

        return self.df