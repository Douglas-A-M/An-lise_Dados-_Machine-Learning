# An-lise_Dados-_Machine-Learning

# Objetivo> Prever o preço de casas baseado em características como nr quartos, localizaç]ao, etc.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Dataset "California Housing" do sklearn
from sklearn.datasets import fetch_california_housing

#Carregar DataSet
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data["Price"] = california.target

#Ánalise Exploratoria
print(data.describe())

#Visualização de correlações
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlação")
plt.show()

#Limpeza
X = data.drop(columns=["Price"])
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Modelo Machine Learning
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error(MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

#Visualização dos Resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="b", label="Valores Reais")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2, label="Valores Previstos")
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Valores Reais vs Valores Previstos")
plt.grid()
plt.show()
