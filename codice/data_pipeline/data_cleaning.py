import pandas as pd


class DataCleaning:
    """Gestisce la pulizia del dataset."""

    def __init__(self, dataframe: pd.DataFrame, is_train: bool = True):
        self.df = dataframe.copy()
        self.is_train = is_train
        self.colonne_eliminate = []

    def pulisci(self) -> pd.DataFrame:
        """Esegue tutte le operazioni di pulizia in sequenza."""
        self.elimina_duplicati()
        self.pulisci_variabili()
        if self.is_train:
            self.elimina_classnull()
        self.rimuovi_outlier_strutturali()
        return self.df

    def elimina_duplicati(self):
        """Rimuove i record duplicati."""
        righe_prima = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicati eliminati: {righe_prima - len(self.df)}. Righe attuali: {len(self.df)}")

    def pulisci_variabili(self):
        """Controlla e corregge i range delle variabili numeriche."""
        print("\n--- CONTROLLO VALORI NUMERICI ---")

        if 'age' in self.df.columns:
            mask_outlier = (self.df['age'] > 800) | (self.df['age'] < 0)
            n_outliers = mask_outlier.sum()
            self.df.loc[mask_outlier, 'age'] = pd.NA
            print(f"Age: {n_outliers} record convertiti in NaN.")

        if 'count_floors_pre_eq' in self.df.columns:
            mask_piani = (self.df['count_floors_pre_eq'] > 15) | (self.df['count_floors_pre_eq'] <= 0)
            self.df.loc[mask_piani, 'count_floors_pre_eq'] = pd.NA
            print(f"Floors: {mask_piani.sum()} valori fuori range corretti.")

        for col in ['area_percentage', 'height_percentage']:
            if col in self.df.columns:
                mask = (self.df[col] <= 0) | (self.df[col] > 100)
                self.df.loc[mask, col] = pd.NA
                print(f"{col}: {mask.sum()} valori fuori range (<=0 o >100) corretti.")

        if 'count_families' in self.df.columns:
            mask_fam = self.df['count_families'] < 0
            self.df.loc[mask_fam, 'count_families'] = pd.NA
            print(f"Families: {mask_fam.sum()} valori negativi corretti.")

    def elimina_classnull(self):
        """Rimuove i record con target nullo."""
        target_col = 'damage_grade'
        righe_originali = self.df.shape[0]
 
        if target_col in self.df.columns:
            self.df = self.df.dropna(subset=[target_col]).reset_index(drop=True)
            righe_dopo = len(self.df)
            print(f"Record con '{target_col}' nullo eliminati: {righe_originali - righe_dopo}")

    def rimuovi_outlier_strutturali(self):
        """Rimuove record che non rispettano i domini di valore attesi."""
        print("\n--- CONTROLLO OUTLIER STRUTTURALI ---")
 
        valid_values = {
            'land_surface_condition': ['n', 'o', 't'],
            'foundation_type': ['h', 'i', 'r', 'u', 'w'],
            'roof_type': ['n', 'q', 'x'],
            'ground_floor_type': ['f', 'm', 'v', 'x', 'z'],
            'other_floor_type': ['j', 's', 'q', 'x'],
            'position': ['j', 's', 'o', 't'],
            'plan_configuration': ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'],
            'legal_ownership_status': ['a', 'r', 'v', 'w'],
        }

        colonne_binarie = [c for c in self.df.columns if c.startswith('has_')]
        for col in colonne_binarie:
            valid_values[col] = [0, 1]
 
        if self.is_train:
            valid_mask = pd.Series(True, index=self.df.index)
 
            if 'geo_level_1_id' in self.df.columns:
                valid_mask &= self.df['geo_level_1_id'].between(0, 30)
            if 'geo_level_2_id' in self.df.columns:
                valid_mask &= self.df['geo_level_2_id'].between(0, 1427)
            if 'geo_level_3_id' in self.df.columns:
                valid_mask &= self.df['geo_level_3_id'].between(0, 12567)
 
            for col, values in valid_values.items():
                if col in self.df.columns:
                    valid_mask &= (self.df[col].isin(values) | self.df[col].isna())
 
            righe_prima = len(self.df)
            self.df = self.df[valid_mask].reset_index(drop=True)
            print(f"[TRAIN] Righe rimosse come outlier: {righe_prima - len(self.df)}")
            print(f"[TRAIN] Righe rimanenti: {len(self.df)}")
        else:
            n_corrections = 0
            for col, values in valid_values.items():
                if col in self.df.columns:
                    mask = ~self.df[col].isin(values) & self.df[col].notna()
                    n_corrections += mask.sum()
                    self.df.loc[mask, col] = pd.NA
            print(f"[TEST] Valori fuori dominio convertiti in NaN: {n_corrections}")
            print(f"[TEST] Righe mantenute (tutte): {len(self.df)}")

    def elimina_record_null_percentuale(self, soglia_percentuale=0.30):
        """Rimuove i record con troppi null."""
        n_colonne = len(self.df.columns)
        min_valori_validi = int(n_colonne * (1 - soglia_percentuale))
 
        righe_prima = len(self.df)
        self.df = self.df.dropna(thresh=min_valori_validi).reset_index(drop=True)
        righe_eliminate = righe_prima - len(self.df)
 
        print(f"Record eliminati per troppi null (Soglia {soglia_percentuale * 100}%): {righe_eliminate}")
 
    def elimina_colonne_nulle(self, soglia_percentuale=0.4):
        """Rimuove le feature che superano la percentuale di null."""
        n_righe_totali = len(self.df)
        limite_nulli = n_righe_totali * soglia_percentuale
 
        serie_nulli = self.df.isnull().sum()
        colonne_da_eliminare = serie_nulli[serie_nulli > limite_nulli].index.tolist()
 
        if colonne_da_eliminare:
            print(f"Eliminate {len(colonne_da_eliminare)} feature (> {soglia_percentuale * 100}% nulli).")
            print(f"Feature rimosse: {colonne_da_eliminare}")
            self.df.drop(columns=colonne_da_eliminare, inplace=True)
            self.colonne_eliminate = colonne_da_eliminare
        else:
            print(f"Controllo Qualità: Tutte le feature rispettano la soglia del {soglia_percentuale * 100}%.")
 
    def applica_colonne_eliminate(self, colonne: list):
        """Elimina le colonne passate."""
        colonne_presenti = [c for c in colonne if c in self.df.columns]
        if colonne_presenti:
            self.df.drop(columns=colonne_presenti, inplace=True)
            print(f"[TEST] Eliminate {len(colonne_presenti)} colonne per allineamento al train.")
 