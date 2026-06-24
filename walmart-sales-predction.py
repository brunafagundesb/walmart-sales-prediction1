import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("data/Walmart.csv")
print(df.head()) #da as primeiras 5 linhas
print(df.shape)  #da o tamanho do dataset
print(df.info()) #verificar se tem dados faltantes

df['Holiday_Flag']= df['Holiday_Flag'].astype(int)
df["Date"] =pd.to_datetime(
    df["Date"], format="%d-%m-%Y"
)
df['Year']= df['Date'].dt.year
df['Month']= df['Date'].dt.month
df['Week']= df['Date'].dt.isocalendar().week

#Histograma das vendas
plt.hist (df['Weekly_Sales'],
        bins=30,
        color= 'purple',
        edgecolor='white',
        alpha=0.6)
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=45)

plt.title("Histograma das vendas")
plt.xlabel('Vendas Semanais ($)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()
plt.savefig("images/histograma_vendas.png")

#vendas por loja
vendas_loja= df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
vendas_loja.plot(kind='bar')
plt.title('Venda Média Semanal por Loja')
plt.xlabel("Loja")
plt.ylabel('Venda Média Semanal ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("images/vendas_por_loja.png")

df = pd.get_dummies(df, columns=['Store'], drop_first=True)

X= df.drop(
    columns=['Weekly_Sales', 'Date']
)
y=df['Weekly_Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

print('Model succesfully trained!')
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)

for var, coef in zip(X.columns, model.coef_):
    print(f"variable: {var}, coeficient: {coef}")

predictions = model.predict(X_test)
r2 =r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse= root_mean_squared_error(y_test, predictions)

plt.scatter(y_test, predictions, alpha=0.5)
plt.title("Valor Real vs Valor Previsto")
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.show()
plt.savefig("images/real_vs_previsto.png")

print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


print(df.corr(numeric_only=True)["Weekly_Sales"].sort_values(ascending=False))

print(X.shape)
print(X.columns)

#para ver quais lojas mais influenciam as vendas
coef_df = pd.DataFrame({
    "Variavel": X.columns,
    "Coeficiente": model.coef_
})

coef_df["Abs"] = coef_df["Coeficiente"].abs()

print(
    coef_df.sort_values(
        by="Abs",
        ascending=False
    ).head(15)
)

#as 10 variaveis mais importantes
#A importancia foi medida pelo valor absoluto do coeficientes da regressão linear.
top_coef =(coef_df.sort_values(by="Abs", ascending=False).head(10).sort_values(by="Coeficiente"))
plt.figure(figsize=(10,6))
plt.barh(top_coef["Variavel"], top_coef["Coeficiente"])
plt.title("As 10 variáveis mais importantes para as vendas")
plt.xlabel("Coeficiente")
plt.ylabel("Variavel")
plt.tight_layout()
plt.show()
plt.savefig("images/variaveis_importantes.png")


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Mapa de Correlação")
plt.show()
plt.savefig("images/mapa_correlacao.png")