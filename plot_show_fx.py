import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier Excel
excel_path = "data.xlsx"  # adapte le chemin si besoin
xls = pd.ExcelFile(excel_path)

# Lire les noms de feuilles
sheet_names = xls.sheet_names

# Charger les données de chaque feuille
dfs = {name: xls.parse(name) for name in sheet_names}

# Créer les subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()

# Tracer chaque signal
for i, name in enumerate(sheet_names):
    df = dfs[name]
    axs[i].scatter(df["X"], df["Y"], s=10, label=name)
    axs[i].set_title(name)
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    axs[i].grid(True)
    axs[i].legend()

# Supprimer la 6e case vide
fig.delaxes(axs[-1])

plt.tight_layout()
plt.show()
