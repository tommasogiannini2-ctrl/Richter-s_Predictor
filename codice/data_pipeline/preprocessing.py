import pandas as pd
from .data_cleaning import DataCleaning
from .data_imputation import DataImputation
from .data_encoding import DataEncoding
from .data_standardization import DataScaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder


class Preprocessing:
    """
    Orchestratore principale della pipeline di preprocessing.
    Coordina in sequenza: pulizia, imputazione, encoding e standardizzazione.
    """

    def __init__(self, dataframe: pd.DataFrame, scaler=None, imputer_num=None, imputer_bin=None, imputer_cat=None, target_encoder=None, colonne_eliminate=None, lista_colonne=None, is_train: bool = True):
        """
        Inizializza l'orchestratore con il dataset, lo stato di training e gli scaler/imputer necessari.

        Input:
          - dataframe (pd.DataFrame): Dataset da preelaborare.
          - scaler (StandardScaler, opzionale): Scaler da iniettare/addestrare.
          - imputer_num, imputer_bin, imputer_cat (SimpleImputer, opzionali): Imputer per i diversi tipi di feature.
          - target_encoder (TargetEncoder, opzionale): Encoder addestrato sulle feature geografiche.
          - colonne_eliminate (list, opzionale): Colonne escluse durante il training.
          - lista_colonne (list, opzionale): Struttura finale delle colonne per l'allineamento.
          - is_train (bool, default=True): Flag che indica se siamo in fase di training o test.

        Output:
          - Nessuno.
        """
        self.df = dataframe.copy()
        self.is_train = is_train

        self.scaler = scaler
        self.imputer_num = imputer_num
        self.imputer_bin = imputer_bin
        self.imputer_cat = imputer_cat
        self.target_encoder = target_encoder
        self.colonne_eliminate = colonne_eliminate if colonne_eliminate is not None else []
        self.lista_colonne = lista_colonne if lista_colonne is not None else []

    def esegui(self) -> pd.DataFrame:
        """
        Esegue la pipeline completa di preprocessing (pulizia, imputazione, encoding, standardizzazione).

        Input:
          - Nessuno.

        Output:
          - pd.DataFrame: Il dataset preelaborato pronto per l'addestramento o la predizione.
        """
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

        target_col = 'damage_grade'
        target_series = None
        if target_col in self.df.columns:
            target_series = self.df[target_col].copy()
 
        # ── FASE 2: IMPUTAZIONE ──────────────────────────────────────
        print(f"\n  [2/4] Imputazione valori mancanti")
        print(f"  {'-' * 48}")
 
        imputation = DataImputation(
            self.df,
            imputer_num=self.imputer_num,
            imputer_bin=self.imputer_bin,   
            imputer_cat=self.imputer_cat,
            is_train=self.is_train,
        )
        self.df = imputation.imputa()
 
        if self.is_train:
            self.imputer_num = imputation.imputer_num
            self.imputer_bin = imputation.imputer_bin
            self.imputer_cat = imputation.imputer_cat

        # ── FASE 2.5: TARGET ENCODING GEOGRAFICO ────────────────────
        geo_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
        geo_cols_presenti = [c for c in geo_cols if c in self.df.columns]

        if geo_cols_presenti:
            print(f"\n  [2.5/4] Target Encoding variabili geografiche")
            print(f"  {'-' * 48}")

            if self.is_train:
                if target_series is None:
                    raise ValueError("TargetEncoder geografico richiede 'damage_grade' nel train set.")

                self.target_encoder = TargetEncoder(
                    categories='auto',
                    target_type='continuous',
                    random_state=42
                )
                encoded_geo = self.target_encoder.fit_transform(
                    self.df[geo_cols_presenti],
                    target_series
                )
                modalita = "fit_transform"
            else:
                if self.target_encoder is None:
                    raise ValueError(
                        "Sul test è obbligatorio passare il TargetEncoder addestrato sul train."
                    )
                encoded_geo = self.target_encoder.transform(self.df[geo_cols_presenti])
                modalita = "transform"

            self.df[geo_cols_presenti] = encoded_geo
            print(f"  {'Colonne geografiche codificate (' + modalita + '):':<40} {len(geo_cols_presenti):>8}")
            print(f"  Colonne: {geo_cols_presenti}")
 
        # ── FASE 3: ENCODING ─────────────────────────────────────────
        print(f"\n  [3/4] Encoding variabili categoriche")
        print(f"  {'-' * 48}")
 
        encoding = DataEncoding(self.df, is_train=self.is_train)
        self.df = encoding.trasforma(
            lista_colonne=self.lista_colonne if not self.is_train else None
        )
 
        # ── FASE 4: STANDARDIZZAZIONE ────────────────────────────────
        print(f"\n  [4/4] Standardizzazione variabili continue")
        print(f"  {'-' * 48}")
 
        scaling = DataScaling(self.df, scaler=self.scaler, is_train=self.is_train)
        self.df = scaling.standardizza()
 
        if self.is_train:
            self.scaler = scaling.scaler
 
        if target_series is not None:
            self.df[target_col] = target_series.values
 
        if self.is_train:
            self.lista_colonne = self.df.columns.tolist()
 
        # ── RIEPILOGO FINALE ─────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  Preprocessing {modalita} completato")
        print(f"  {'Righe:':<40} {self.df.shape[0]:>8}")
        print(f"  {'Colonne:':<40} {self.df.shape[1]:>8}")
        print(f"  {'Valori mancanti residui:':<40} {self.df.isnull().sum().sum():>8}")
        print(f"{'=' * 60}\n")
 
        return self.df


def dividi_train_validation_test(df, target_column='damage_grade'):
    """
    Suddivide il dataset originario in Train (70%), Validation (15%) e Test (15%) in modo stratificato.

    Input:
      - df (pd.DataFrame): Dataset complessivo da partizionare.
      - target_column (str, default='damage_grade'): Colonna target da usare per la stratificazione.

    Output:
      - tuple: Contiene (train_df, val_df, test_df).
    """
    # Prepariamo la stratificazione se la colonna è fornita
    strat = df[target_column] if target_column else None

    # STEP 1: 70% Train, 30% Temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        shuffle=True,
        stratify=strat
    )

    # Prepariamo la stratificazione per il secondo split
    strat_temp = temp_df[target_column] if target_column else None

    # STEP 2: 15% Validation, 15% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
        stratify=strat_temp
    )

    print(f"Training set:   {len(train_df)} ({len(train_df) / len(df):.0%})")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(df):.0%})")
    print(f"Test set:       {len(test_df)} ({len(test_df) / len(df):.0%})")

    return train_df, val_df, test_df
