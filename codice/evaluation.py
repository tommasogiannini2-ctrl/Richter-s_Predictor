import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize


# --- LA TUA CLASSE ---
class ModelEvaluator:
    def __init__(self, y_true, y_pred, y_score, target_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score
        self.classes = np.unique(y_true)
        self.target_names = target_names if target_names else [f'Classe {c}' for c in self.classes]

    def calcola_metriche(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        tpr_per_classe = []
        tnr_per_classe = []

        print("\n" + "=" * 50)
        print(f"{'CLASSE':<12} | {'TPR (Rec)':<10} | {'TNR (Spec)':<10}")
        print("-" * 50)

        for i, cls in enumerate(self.classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)

            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
            tpr_per_classe.append(tpr)
            tnr_per_classe.append(tnr)
            print(f"{self.target_names[i]:<12} | {tpr:<10.4f} | {tnr:<10.4f}")

        avg_tpr = np.mean(tpr_per_classe)
        avg_tnr = np.mean(tnr_per_classe)
        g_mean = np.sqrt(avg_tpr * avg_tnr)

        print("-" * 50)
        print(f"GEOMETRIC MEAN: {g_mean:.4f}")
        print("=" * 50)

    def plot_grafici(self):
        plt.figure(figsize=(12, 5))

        # 1. Matrice di Confusione
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names,
                    yticklabels=self.target_names)

        # Etichette degli assi
        plt.title('Matrice di Confusione')
        plt.ylabel('Valori Reali')  # Y = True
        plt.xlabel('Valori Predetti')  # X = Predicted

        # 2. Curva ROC
        plt.subplot(1, 2, 2)
        y_true_bin = label_binarize(self.y_true, classes=self.classes)
        for i in range(len(self.classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_score[:, i])
            plt.plot(fpr, tpr, label=f'{self.target_names[i]} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title('Curve ROC per Classe')
        plt.xlabel('False Positive Rate (1 - Specificità)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


