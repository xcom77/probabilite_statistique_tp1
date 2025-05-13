import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === 1. Charger les données ===
df = pd.read_excel("data.xlsx", sheet_name="Signal_1")
X = df["X"].values
Y = df["Y"].values

# Trier les données
sorted_idx = np.argsort(X)
X = X[sorted_idx]
Y = Y[sorted_idx]

# === 2. Couper les données en 2 moitiés approximatives ===
mid = 40
X_left, Y_left = X[:mid].reshape(-1, 1), Y[:mid]
X_right, Y_right = X[mid:].reshape(-1, 1), Y[mid:]

# === 3. Régressions linéaires sur chaque moitié ===
model_left = LinearRegression().fit(X_left, Y_left)
model_right = LinearRegression().fit(X_right, Y_right)

E = model_left.coef_[0]
b = model_left.intercept_
H = model_right.coef_[0]
b2 = model_right.intercept_

# === 4. Calcul de l'intersection ===
Xc = (b2 - b) / (E - H)

Y0 = E * Xc + b 

# === 5. Découpage autour de Xc ===
mask_left = X <= Xc
mask_right = X > Xc

X_left_cut = X[mask_left].reshape(-1, 1)
Y_left_cut = Y[mask_left]
X_right_cut = X[mask_right].reshape(-1, 1)
Y_right_cut = Y[mask_right]

# Prédictions
Y_pred_left = model_left.predict(X_left_cut)
Y_pred_right = model_right.predict(X_right_cut)

b1 = (Y_left_cut - E*X_left_cut.flatten())

mean_b = np.mean(b1)
std_b = np.std(b1, ddof=1)  

b2 = (Y_right_cut - H*X_right_cut.flatten() - Y0) 

mean_b2 = np.mean(b2)
std_b2 = np.std(b2, ddof=1)



# === 7. Affichage du graphique ===
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', s=10, label='Données')
plt.plot(X_left_cut, Y_pred_left, color='red', label='Régression gauche')
plt.plot(X_right_cut, Y_pred_right, color='green', label='Régression droite')
plt.axvline(Xc, color='gray', linestyle='--', label=f"Xc = {Xc:.4f}")
plt.title("Signal_1 – Modèle C (intersection)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. Résultats ===
print("===== PARAMÈTRES AVEC INTERSECTION =====")
print(f"E        = {E:.4f}")
print(f"H        = {H:.4f}")
print(f"Y0       = {Y0:.4f}")
print(f"Xc       = {Xc:.4f}")
print(f"mean(b1) = {mean_b:.4f}")
print(f"std(b1)  = {std_b:.4f}")

print(f"mean(b1) = {mean_b2:.4f}")
print(f"std(b1)  = {std_b2:.4f}")



