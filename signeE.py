import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

# === 1. Charger les données ===
df = pd.read_excel("data.xlsx", sheet_name="Signal_3")
X = df["X"].values
Y = df["Y"].values

# === 2. Définir les modèles ===
def model_tanh(x, A, B):
    return A * np.tanh(B * x)

def model_arctan(x, A, B):
    return A * np.arctan(B * x)

def model_erf(x, A, B):
    return A * erf(B * x)

# === 3. Ajustements ===
p0 = [1, 5]  # Guess: A=1, B=5

params_tanh, _ = curve_fit(model_tanh, X, Y, p0=p0)
params_arctan, _ = curve_fit(model_arctan, X, Y, p0=p0)
params_erf, _ = curve_fit(model_erf, X, Y, p0=p0)

# === 4. Prédictions ===
x_fit = np.linspace(min(X), max(X), 300)
y_tanh = model_tanh(x_fit, *params_tanh)
y_arctan = model_arctan(x_fit, *params_arctan)
y_erf = model_erf(x_fit, *params_erf)

# === 5. Résultats ===
print("=== Tanh ===")
print(f"A = {params_tanh[0]:.4f}, B = {params_tanh[1]:.4f}")
print("=== Arctan ===")
print(f"A = {params_arctan[0]:.4f}, B = {params_arctan[1]:.4f}")
print("=== Erf ===")
print(f"A = {params_erf[0]:.4f}, B = {params_erf[1]:.4f}")

# === 6. Visualisation ===
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, s=15, label="Données")
plt.plot(x_fit, y_tanh, label="Modèle tanh", color="green")
plt.plot(x_fit, y_erf, label="Modèle erf", color="blue", linestyle=":")
plt.title("Ajustements symétriques sur Signal_3")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
