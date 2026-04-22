import pandas as pd
from sklearn.model_selection import train_test_split


class DataReducer:
    """
    Classe per la gestione della pesantezza del dataset.
    Permette di campionare i dati in base all'occupazione di memoria,
    mantenendo le proporzioni del target costanti (campionamento stratificato).
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.target = 'damage_grade'

    def get_info(self):
        """Restituisce numero di record e memoria occupata in MB."""
        n_record = len(self.df)
        memoria_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        return n_record, memoria_mb

    def riduci_per_memoria(self, limite_mb: float):
        """
        Calcola la frazione necessaria per far rientrare il dataset nel limite
        di MB e applica il campionamento stratificato.
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

    def interfaccia_utente(self):
        """Gestisce il dialogo con l'utente per la riduzione."""
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