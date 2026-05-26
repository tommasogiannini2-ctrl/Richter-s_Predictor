import pandas as pd
from sklearn.model_selection import train_test_split


class DataReducer:
    """
    Gestione del campionamento stratificato per la riduzione del dataset.
    Permette di ridurre l'occupazione di memoria del dataset salvaguardando
    le proporzioni originarie delle classi del target.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Inizializza la classe predisponendo il dataframe e identificando la colonna target.

        Input:
          - dataframe (pd.DataFrame): Dataset da sottoporre a campionamento.

        Output:
          - Nessuno.
        """
        self.df = dataframe.copy()
        self.target = 'damage_grade'

    def get_info(self) -> tuple:
        """
        Calcola e restituisce il numero di record e la dimensione in memoria del dataset.

        Input:
          - Nessuno.

        Output:
          - tuple: Contiene (numero_di_righe, dimensione_in_megabyte).
        """
        n_record = len(self.df)
        memoria_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        return n_record, memoria_mb

    def riduci_per_memoria(self, limite_mb: float) -> pd.DataFrame:
        """
        Esegue il campionamento stratificato per far rientrare il dataset nel limite di memoria indicato.

        Input:
          - limite_mb (float): Valore massimo desiderato in MB per il dataset finale.

        Output:
          - pd.DataFrame: Dataset ridotto.
        """
        n_record, memoria_attuale = self.get_info()

        if memoria_attuale <= limite_mb:
            print(f"  Il dataset occupa già {memoria_attuale:.2f} MB: nessuna riduzione necessaria.")
            return self.df

        frazione = limite_mb / memoria_attuale
        print(f"  {'Frazione mantenuta:':<40} {frazione * 100:>7.1f}%")
        print(f"  {'Limite impostato:':<40} {limite_mb:>7.1f} MB")

        df_ridotto, _ = train_test_split(
            self.df,
            train_size=frazione,
            stratify=self.df[self.target] if self.target in self.df.columns else None,
            random_state=42
        )

        self.df = df_ridotto.reset_index(drop=True)
        return self.df

    def interfaccia_utente(self) -> pd.DataFrame:
        """
        Gestisce l'interazione testuale con l'utente per decidere e configurare la riduzione del dataset.

        Input:
          - Nessuno.

        Output:
          - pd.DataFrame: Il dataset (eventualmente ridotto).
        """
        n, mem = self.get_info()

        print(f"\n{'=' * 60}")
        print(f"  ANALISI DIMENSIONI DATASET")
        print(f"{'=' * 60}")
        print(f"  {'Record totali:':<40} {n:>8}")
        print(f"  {'Memoria occupata:':<40} {mem:>7.2f} MB")
        print(f"{'=' * 60}")

        scelta = input("\n  La dimensione del dataset è ottimale? (s/n): ").strip().lower()

        if scelta == 'n':
            try:
                limite = float(input("  Limite massimo in MB: "))
                print()
                self.df = self.riduci_per_memoria(limite)
                n_nuovo, mem_nuova = self.get_info()
                print(f"  {'Record dopo la riduzione:':<40} {n_nuovo:>8}")
                print(f"  {'Memoria dopo la riduzione:':<40} {mem_nuova:>7.2f} MB")
            except ValueError:
                print("  Valore non valido: procedo senza riduzioni.")
        else:
            print("  Nessuna riduzione applicata.")

        print(f"{'=' * 60}\n")
        return self.df