import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .config import OUTPUTS_DIR

def plot_confusion_metrices(results, class_names):
    OUTPUTS_DIR.mkdir(parents = True, exist_ok=True)
    
    for name, metrics in results.items():
        cm = metrics["confusion_matrix"]
        
        fig, ax = plt.subplots(figsize=(10,8))
        
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        
        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False,
            xticks_rotation=45
        )
        
        ax.set_title(f"{name} confusion Matrix")
        
        fig.tight_layout()
        
        filename = f"{name.lower()}_confusion-matrix.png"
        fig.savefig(OUTPUTS_DIR / filename, dpi = 300)
        
        plt.show()
        plt.close(fig)
