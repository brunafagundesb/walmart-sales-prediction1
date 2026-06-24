# Previsão de Vendas Semanais do Walmart com Regressão Linear

## Objetivo
Este projeto tem como objetivo construir um modelo capaz de prever as vendas semanais das lojas Walmart utilizando variáveis econômicas, climáticas e temporais.

## Dataset
Este dataset contém:

- 6.435 registros

- 45 lojas Walmart

- Dados do período entre 2010 e 2012

Variáveis         
- Store              
- Date               
- Weekly_Sales       
- Holiday_Flag       
- Temperature        
- Fuel_Price         
- CPI                
- Unemployment       

## Programas utilizados
- Python

- Pandas

- NumPy

- Matplotlib

- Seaborn

- Scikit-Learn


## Etapas
1. Importação dos dados
2. Análise Exploratória de Dados (EDA)
foram avaliadas inicialmente:
- Estrutura dos dados
- Tipos das variáveis
- Valores faltantes
- Correlação entre as variáveis
3. Limpeza dos dados 
4. Feature Engineering e One-Hot Encoding 
foram criadas novas variáveis a partir da coluna Date (Year, Month, Week), que permitiram entender possíveis padrões sazonais. Também houve o tratamento da variável Store
5. Treinamento do modelo
6. Avaliação de resultados
7. Melhoramento do modelo

## Observações

Inicialmente, a variável Store foi utilizada como numérica, resultando em um desempenho insatisfatório do modelo(R²=0.15). O algoritmo interpretou as lojas como valores quantitativos, assumindo incorretamente relações lineares entre elas. Ao identificar que Store é uma variável categórica, aplicou-se One-Hot Encoding, gerando 44 variáveis binárias. Isso permitiu que o modelo entendesse as diferenças específicas entre as lojas, aumentando o desempenho para R²=0.93.

## Principais Insights
-A identidade da loja foi o fator mais relevante para explicar as vendas.

-Variáveis econômicas como CPI e desemprego apresentam baixa correlação com o faturamento semanal.

-O desempenho do modelo dependeu fortemente do tratamento correto das variáveis categóricas
 

1. Store_20 -> 0.283

2. Store_4 ->  0.280

3. Store_14 -> 0.260

O que mostra que a identidade e características específicas de cada loja pode ser um fator mais importante para explicar as vendas do que os fatores econômicos presentes no dataset.

## Modelo

Foi usado uma Regressão Linear do Scikit-Learn

Divisão dos dados:

test_size=0.2

random_state=42

## Resultados após One-Hot Encoding 

R² = 0.9256

RMSE = 154.855

MSE = 23.98 bilhões

O modelo foi capaz de explicar aproxidamente 92,5% da variação das vendas semanais, indicando uma ótima capacidade preditiva para os dados disponíveis.

## Conclusões

O projeto demonstrou a importância de Feature Engineering em Machine Learning. A transformação da variável Store por meio de One-Hot Encoding aumentou significativamente o desempenho do modelo, passando de 0.15 para 0.93, sem a necessidade de usar algoritmos mais complexos. A representação correta dos dados é tão importante quanto a escolha do modelo. 


## Histograma das Vendas
![Histograma](images/histograma_vendas.png)

## Venda Média por Loja
![Vendas por Loja](images/vendas_por_loja.png)

## Valores Reais vs Previstos
![Real vs Previsto](images/real_vs_previsto.png)

## Variáveis mais influentes
![10 variáveis mais influentes](images/variaveis_importantes.png)

## Mapa de correlação
![Mapa de correlação](images/mapa_correlacao.png)