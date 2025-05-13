import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

excel_path = "data.xlsx"  
df_A = pd.read_excel(excel_path, sheet_name="Signal_4")

X = df_A["X"].values.reshape(-1, 1)
Y = df_A["Y"].values

model = LinearRegression()
model.fit(X, Y)

E = model.coef_[0]
b0 = model.intercept_
print(f" Coefficient E = {E:.4f}")
print(f" Ordonnée à l'origine b0 = {b0:.4f}")

Y_pred = model.predict(X)

plt.scatter(X, Y, label="Données", color="blue", s=10)
plt.plot(X, Y_pred, label=f"Régression: Y = {E:.2f}X + {b0:.2f}", color="red")
plt.title("Signal A – Régression linéaire")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()

X_2 = X
b_tab = []
for i in range(len(Y)):
    b_tab.append(Y[i] - E * X[i])
b_mean = sum(b_tab)/len(b_tab)

print(b_mean)

b_std = np.std(b_tab)

print(b_std)