import pandas as pd
import os
from abc import ABC, abstractmethod
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# --- 1. DEFINIZIONE DELLA FACTORY ---
class AbstractOpener(ABC):
    def open(self, dataframe_path: str) -> pd.DataFrame:
        if not os.path.exists(dataframe_path):
            raise FileNotFoundError(f"File {dataframe_path} non trovato")
        try:
            return self._load_data(dataframe_path)
        except Exception as e:
            raise RuntimeError(f"Errore durante la lettura del file {dataframe_path}: {e}")

    @abstractmethod
    def _load_data(self, path: str) -> pd.DataFrame:
        pass

class XLSOpener(AbstractOpener):
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_excel(path)

class CSVOpener(AbstractOpener):
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

class JSONOpener(AbstractOpener):
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_json(path)

def scegli_opener(dataframe_path: str) -> AbstractOpener:
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

# --- 2. CLASSE PREPROCESSING ---
class Preprocessing:
    """
    Classe incaricata della pulizia e preparazione del dataset.
    Riceve il dataframe già unito e restituisce il dataframe processato.
    """
    def __init__(self, dataframe: pd.DataFrame, scaler=None, lista_colonne=None, is_train = True, choice=2):
        self.df = dataframe.copy()
        # Permette di passare uno scaler pre-fittato (utile quando si processerà il test set)
        self.scaler = scaler if scaler else StandardScaler()
        self.lista_colonne = lista_colonne if lista_colonne else []
        self.is_train = is_train

        if choice not in [1, 2, 3]:
            raise ValueError("Il parametro 'choice' deve essere 1 (rimozione), 2 (media) o 3 (KNN).")
        self.choice = choice


    def esegui(self) -> pd.DataFrame:
        print(f"\nAvvio Preprocessing ({'Train' if self.is_train else 'Test'})...")

        self.elimina_duplicati()
        self.pulisci_variabili()
        self.elimina_classnull()

        if self.is_train:
            self.elimina_record_null_percentuale()
            self.elimina_colonne_nulle()
            self.gestisci_valori_mancanti()
            self.dummy()

            self.lista_colonne = self.df.columns.tolist()

        else:
            self.gestisci_valori_mancanti()
            self.dummy()

            self.df = self.df.reindex(columns=self.lista_colonne, fill_value=0)

        self.standardizza(self.is_train)

        return self.df

    # Metodo per eliminare duplicati
    def elimina_duplicati(self):
        self.df = self.df.drop_duplicates()
        # Riassegna gli indici dopo l'eliminazione
        self.df = self.df.reset_index(drop=True)


    def gestisci_valori_mancanti_rimozione(self):
        da = self.df.copy()
        da.dropna(inplace=True)
        da.reset_index(drop=True, inplace=True)
        print("Righe con valori mancanti eliminate con successo!")
        self.df = da


    def gestisci_valori_mancanti_media(self):
        # Imputazione con la media su tutte le colonne numeriche che hanno NaN
        da = self.df.copy()
        numeric_cols_with_nan = da.select_dtypes(include='number').columns[
            da.select_dtypes(include='number').isnull().any()]
        if self.is_train:
            for col in numeric_cols_with_nan:
                col_mean = da[col].mean()
                da[col] = da[col].fillna(col_mean)
            print(f"Imputazione univariata eseguita su colonne di train: {list(numeric_cols_with_nan)}")

        else:
            i =0
            for col in numeric_cols_with_nan:
                col_mean = self.scaler.mean_[i]
                da[col] = da[col].fillna(col_mean)
                i = +1
            print(f"Imputazione univariata eseguita su colonne di test: {list(numeric_cols_with_nan)}")
        self.df = da

    def gestisci_valori_mancanti_KNN(self):
        da = self.df.copy()
        numeric_cols = da.select_dtypes(include='number').columns
        imputer = KNNImputer(n_neighbors=5)
        da[numeric_cols] = imputer.fit_transform(da[numeric_cols])
        print("Imputazione multivariata (KNN) eseguita con successo!")
        self.df = da


    def gestisci_valori_mancanti(self):
        """Gestione interattiva dei valori nulli (NaN)."""

        n = self.df.isnull().sum().sum()
        print(f"\nSono stati trovati {n} valori mancanti.")

        if n>0:

            if self.choice == 1:
                print("Scelta 1. Eliminazione del record")
                self.gestisci_valori_mancanti_rimozione()


            elif self.choice == 2:
                print("Scelta 2. Imputazione univariata (media per ogni colonna numerica)")
                self.gestisci_valori_mancanti_media()


            elif self.choice == 3:
                print("Scelta 3. Imputazione multivariata (KNN Imputer)")
                self.gestisci_valori_mancanti_KNN()

            else:
                print("Scelta non valida. Uscita senza modifiche.")

            #Controllo finale sui nulli
            n = self.df.isnull().sum().sum()
            print(f"\nSono stati trovati {n} valori mancanti dopo la pulizia.")

        else:
            print("Nessun valore mancante trovato")


    def dummy(self):
        """
        Trasforma le feature categoriche in dummy variables .
        """
        # Lista delle feature categoriche del dataset originale
        feature_categoriche = [
            'land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position',
            'plan_configuration', 'legal_ownership_status'
        ]
        # drop_first=False crea una colonna per OGNI categoria
        # dtype=int serve per avere 0/1 invece di True/False
        self.df = pd.get_dummies(self.df, columns=feature_categoriche, drop_first=False, dtype=int)

        # Calcoliamo quante nuove feature sono state generate
        nuove_colonne = [col for col in self.df.columns if any(feat in col for feat in feature_categoriche) and col not in feature_categoriche]
        print(f"Aggiunte {len(nuove_colonne)} nuove colonne dummy.")

    def pulisci_variabili(self):
        """
        Controlla e corregge i range delle variabili numeriche continue.
        """
        print("\n--- CONTROLLO VALORI NUMERICI ---")

        if 'age' in self.df.columns:
            mask_outlier = (self.df['age'] > 800) | (self.df['age'] < 0)
            n_outliers = mask_outlier.sum()

            # Sostituiamo con NaN per l'imputazione successiva
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

    def rimuovi_outlier_strutturali(self):
        """
        Rimuove record che non rispettano i domini di valore attesi.
        Utile per pulire errori di inserimento dati.
        """
        print("\n--- CONTROLLO OUTLIER ---")

        # Valori attesi per ogni colonna categorica/binaria
        valid_values = {
            'land_surface_condition': ['n', 'o', 't'],
            'foundation_type': ['h', 'i', 'r', 'u', 'w'],
            'roof_type': ['n', 'q', 'x'],
            'ground_floor_type': ['f', 'm', 'v', 'x', 'z'],
            'other_floor_type': ['j', 's', 'q', 'x'],
            'position': ['j', 's', 'o', 't'],
            'plan_configuration': ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'],
            'legal_ownership_status': ['a', 'r', 'v', 'w'],
            'has_superstructure_adobe_mud': [0, 1],
            'has_superstructure_mud_mortar_stone': [0, 1],
            'has_superstructure_stone_flag': [0, 1],
            'has_superstructure_cement_mortar_stone': [0, 1],
            'has_superstructure_mud_mortar_brick': [0, 1],
            'has_superstructure_cement_mortar_brick': [0, 1],
            'has_superstructure_timber': [0, 1],
            'has_superstructure_rc_non_engineered': [0, 1],
            'has_superstructure_rc_engineered': [0, 1],
            'has_superstructure_other': [0, 1],
            'has_secondary_use': [0, 1],
            'has_secondary_use_agriculture': [0, 1],
            'has_secondary_use_hotel': [0, 1],
            'has_secondary_use_rental': [0, 1],
            'has_secondary_use_institution': [0, 1],
            'has_secondary_use_school': [0, 1],
            'has_secondary_use_industry': [0, 1],
            'has_secondary_use_health_post': [0, 1],
            'has_secondary_use_gov_office': [0, 1],
            'has_secondary_use_use_police': [0, 1],
            'has_secondary_use_other': [0, 1],
        }

        # Costruisci una maschera booleana: True = riga valida
        valid_mask = (
                self.df['geo_level_1_id'].between(0, 30) &
                self.df['geo_level_2_id'].between(0, 1427) &
                self.df['geo_level_3_id'].between(0, 12567)
        )

        for col, values in valid_values.items():
            if col in self.df.columns:
                valid_mask &= (self.df[col].isin(values) | self.df[col].isna())

        righe_prima = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        righe_dopo = len(self.df)

        print(f"Righe rimosse come outlier: {righe_prima - righe_dopo}")
        print(f"Righe rimanenti: {righe_dopo}")

    def elimina_classnull(self):
        target_col = 'damage_grade'
        righe_originali = self.df.shape[0]
        if target_col in self.df.columns:
            # Rimuove le righe dove il valore nella colonna 'damage_grade' è nullo (NaN)
            self.df = self.df.dropna(subset=[target_col]).reset_index(drop=True)
            righe_dopo_aver_tolto_i_null = len(self.df)
            assert righe_dopo_aver_tolto_i_null <= righe_originali, "ERRORE: le righe dopo la pulizia sono aumentate!"

    def elimina_record_null_percentuale(self, soglia_percentuale=0.30):
        """
        Rimuove i record che hanno una percentuale di valori nulli superiore alla soglia.
        soglia_percentuale=0.30.
        """
        n_colonne = len(self.df.columns)
        min_valori_validi = int(n_colonne * (1 - soglia_percentuale))

        dati_puliti = self.df.dropna(thresh=min_valori_validi).reset_index(drop=True)

        righe_eliminate = len(self.df) - len(dati_puliti)
        self.df = dati_puliti
        print(f"Record eliminati per troppi null sulla riga (Soglia {soglia_percentuale * 100}%): {righe_eliminate}")

    def elimina_colonne_nulle(self,soglia_percentuale=0.4):
        """
        Rimuove le feature che superano la percentuale di valori nulli indicata.
        Soglia 0.4 = Elimina se mancano più del 40% dei dati.
        """
        #  Calcolo del numero massimo di nulli consentiti per colonna
        n_righe_totali = len(self.df)
        limite_nulli = n_righe_totali * soglia_percentuale

        # Identificazione delle colonne che superano il limite
        serie_nulli = self.df.isnull().sum()
        colonne_da_eliminare = serie_nulli[serie_nulli > limite_nulli].index.tolist()

        # Log dei risultati (utile per la tua relazione FIA)
        if colonne_da_eliminare:
            print(
                f"Attenzione: Eliminate {len(colonne_da_eliminare)} feature (> {soglia_percentuale * 100}% nulli).")
            print(f"Feature rimosse: {colonne_da_eliminare}")
            # Restituiamo il dataframe senza quelle colonne
            self.df.drop(columns=colonne_da_eliminare, inplace=True)
        else:
            print(f"Controllo Qualità: Tutte le feature rispettano la soglia del {soglia_percentuale * 100}%.")

    def standardizza(self, is_train=True):
        """
        Standardizza esclusivamente le feature numeriche continue (evitando di modificare ID o variabili categoriche).
        """
        print("\n--- STANDARDIZZAZIONE ---")
        # Identifichiamo le variabili continue previste dal problema
        colonne_continue = [
            'age', 
            'area_percentage', 
            'height_percentage', 
            'count_floors_pre_eq', 
            'count_families'
        ]
        
        # Filtriamo le colonne per accertarci che siano presenti nel dataframe
        colonne_da_standardizzare = [col for col in colonne_continue if col in self.df.columns]
        
        if not colonne_da_standardizzare:
            print("Nessuna colonna continua trovata per la standardizzazione.")
            return

        if is_train:
            # Per i dati di Train calcoliamo (fit) e applichiamo (transform) la standardizzazione
            self.df[colonne_da_standardizzare] = self.scaler.fit_transform(self.df[colonne_da_standardizzare])
            print(f"Scaler media di train : {self.scaler.mean_}")
            print(f"Standardizzazione calcolata e applicata (fit_transform) su: {colonne_da_standardizzare}")
        else:
            # Per i dati di Test applichiamo (transform) le metriche calcolate precedentemente sul Train per evitare Data Leakage
            self.df[colonne_da_standardizzare] = self.scaler.transform(self.df[colonne_da_standardizzare])
            print(f"Standardizzazione applicata (transform) su: {colonne_da_standardizzare}")
            
        print("\nEsempio di valori standardizzati (prime 5 righe delle colonne modificate):")
        print(self.df[colonne_da_standardizzare].head())