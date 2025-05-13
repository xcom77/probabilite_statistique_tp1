import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


excel_path = "data.xlsx"
df = pd.read_excel(excel_path, sheet_name="Signal_5")

X = df["X"].values.reshape(-1, 1)
Y = df["Y"].values

model = LinearRegression()
model.fit(X, Y)

E = model.coef_[0]
b0 = model.intercept_

b = (Y / (E * X.flatten())) - 1

mean_b = np.mean(b)
std_b = np.std(b, ddof=1)  

Y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label="Données Signal_5", color="blue", s=20)
plt.plot(X, Y_pred, label=f"Régression linéaire", color="red", linewidth=2)

text = (
    f"E = {E:.2f}\n"
    f"mean(b) = {mean_b:.2e}\n"
    f"std(b) = {std_b:.2e}"
)
plt.text(0.01, max(Y) * 0.7, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.title("Signal_5 – Régression linéaire avec bruit multiplicatif")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


print(mean_b)
print(std_b)