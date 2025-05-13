import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === 1. Charger les données ===
df = pd.read_excel("data.xlsx", sheet_name="Signal_2")
X = df["X"].values
Y = df["Y"].values

# === 2. Définir Xc ===
Xc = 0.23

# === 3. Partie gauche : X < Xc ===
mask_left = X < Xc
X_left = X[mask_left].reshape(-1, 1)
Y_left = Y[mask_left]

linreg = LinearRegression().fit(X_left, Y_left)
E = linreg.coef_[0]
b1 = linreg.intercept_
Y_pred_left = linreg.predict(X_left)
resid_b1 = Y_left - Y_pred_left
mean_b1 = np.mean(resid_b1)
std_b1 = np.std(resid_b1)

# === 4. Partie droite : X >= Xc ===
mask_right = X >= Xc
X_right = X[mask_right]
Y_right = Y[mask_right]

# Modèle non-linéaire
def model_right(x, H, n, Y0, b2):
    return H * (x - Xc)**n + Y0 + b2

# Valeurs initiales approximatives
initial_guess = [1000, 1, 500, 0]

params_opt, _ = curve_fit(model_right, X_right, Y_right, p0=initial_guess)
H, n, Y0, b2 = params_opt
Y_pred_right = model_right(X_right, *params_opt)
resid_b2 = Y_right - Y_pred_right
mean_b2 = np.mean(resid_b2)
std_b2 = np.std(resid_b2)

# === 5. Résultats ===
print("=== Résultats pour Signal_2 ===")
print(f"E        = {E:.4f}")
print(f"mean(b1) = {mean_b1:.4f}, std(b1) = {std_b1:.4f}")
print(f"Xc       = {Xc:.4f}")
print(f"H        = {H:.4f}")
print(f"n        = {n:.4f}")
print(f"Y0       = {Y0:.4f}")
print(f"mean(b2) = {mean_b2:.4f}, std(b2) = {std_b2:.4f}")

# === 6. Visualisation ===
plt.scatter(X, Y, s=10, label="Données")
plt.plot(X_left, Y_pred_left, color="green", label="Fit linéaire (X < Xc)")
plt.plot(X_right, Y_pred_right, color="red", label="Fit non-linéaire (X ≥ Xc)")
plt.axvline(Xc, color='gray', linestyle='--', label=f"Xc = {Xc}")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ajustement du Signal_2")
plt.legend()
plt.grid(True)
plt.show()
