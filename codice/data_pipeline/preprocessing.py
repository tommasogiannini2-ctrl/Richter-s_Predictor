import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importa i moduli della pipeline
from .data_cleaning import DataCleaning
from .data_imputation import DataImputation
from .data_encoding import DataEncoding
from .data_standardization import DataScaling


class Preprocessing:
    """
    Orchestratore principale della pipeline di preprocessing.
    Coordina pulizia, imputazione ed encoding del dataset.
    """
    def __init__(self, dataframe: pd.DataFrame, scaler=None, imputation_models=None, lista_colonne=None, is_train=True):
        self.df = dataframe.copy()
        self.scaler = scaler
        self.imputation_models = imputation_models
        self.lista_colonne = lista_colonne
        self.is_train = is_train
        self.imputatore_istanza = None

    def esegui(self) -> pd.DataFrame:
        """Esegue l'intera pipeline di preprocessing."""
        print(f"\n{'=' * 60}")
        print(f"Avvio Preprocessing ({'Train' if self.is_train else 'Test'})...")
        print(f"{'=' * 60}")

        # FASE 1: PULIZIA
        print(f"\n[FASE 1/4] Pulizia dei dati...")
        cleaning = DataCleaning(self.df)

        if self.is_train:
            cleaning.elimina_record_null_percentuale()
            cleaning.elimina_colonne_nulle()

        self.df = cleaning.pulisci()

        # FASE 2: ENCODING DUMMY
        print(f"\n[FASE 2/4] Encoding delle variabili categoriche (Dummy)...")
        encoder = DataEncoding(self.df)
        self.df = encoder.dummy()

        # Allineamento colonne per il Test Set
        if not self.is_train and self.lista_colonne:
            # Assicuriamoci che il test set abbia le stesse colonne del train (riempiendo con 0 le mancanti)
            self.df = self.df.reindex(columns=self.lista_colonne, fill_value=0)

        # FASE 3: IMPUTAZIONE MULTIVARIATA
        print(f"\n[FASE 3/4] Imputazione multivariata dei valori mancanti...")
        imputation = DataImputation(is_train=self.is_train, models=self.imputation_models)
        self.df = imputation.imputa(self.df)
        self.imputatore_istanza = imputation

        # FASE 4: STANDARDIZZAZIONE
        print(f"\n[FASE 4/4] Standardizzazione delle feature continue...")
        scaler_obj = DataScaling(self.df, self.scaler, self.is_train)
        self.df = scaler_obj.standardizza()
        self.scaler = scaler_obj.scaler

        # Aggiornamento della lista di colonne per il test set
        if self.is_train:
            self.lista_colonne = self.df.columns.tolist()

        print(f"\n{'=' * 60}")
        print(f"Preprocessing completato! Dataset shape: {self.df.shape}")
        print(f"{'=' * 60}\n")

        return self.df
