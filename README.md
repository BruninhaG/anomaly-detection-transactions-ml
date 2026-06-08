# Detecção de Anomalias em Transações Financeiras usando Machine Learning

## 📌 Sobre o Projeto
A identificação de anomalias em fluxos de transações é um pilar crítico para a prevenção de fraudes, mitigação de riscos e auditoria financeira. Este projeto explora duas abordagens distintas e complementares para detectar desvios em dados transacionais: **Modelagem Estatística de Séries Temporais (ARIMA)** e **Aprendizado de Máquina Não Supervisionado (Isolation Forest)**.

## 🧠 Abordagens Implementadas

### 1. Modelo ARIMA (Autoregressive Integrated Moving Average)
Focado na análise do comportamento histórico dos dados ao longo do tempo. O modelo aprende o padrão cíclico das transações e aponta como anomalia qualquer desvio significativo fora das bandas de predição esperadas.

* **Ideal para:** Identificar variações volumétricas atípicas em linhas do tempo.
* **Métricas Principais:** Termos Autorregressivos ($AR$) e de Média Móvel ($MA$).

### 2. Algoritmo Isolation Forest
Um algoritmo de Machine Learning baseado em árvores de decisão, projetado especificamente para isolar anomalias em vez de perfilar pontos normais. Como as anomalias possuem atributos distintos e são raras, elas são isoladas mais rapidamente nas estruturas da árvore.

* **Ideal para:** Identificar transações fraudulentas isoladas com valores fora do padrão (outliers).
* **Métrica Principal:** Taxa de contaminação (`contamination`).

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Python 3.10+**
* **Pandas:** Manipulação, limpeza e indexação temporal de dados.
* **Scikit-Learn:** Implementação do modelo robusto de *Isolation Forest*.
* **PyFlux:** Modelagem estatística avançada para o *ARIMA*.
* **Matplotlib:** Visualização gráfica dos ajustes do modelo e anomalias.

---

## 🚀 Exemplos de Implementação

### 🔹 Abordagem 1: Modelagem Temporal com ARIMA

```python
import pandas as pd
import pyflux as pf
import matplotlib.pyplot as plt

# Carga dos dados com indexação temporal
data = pd.read_csv('transaction_data.csv', parse_dates=['datetime'], index_col='datetime')

# Ajuste do modelo ARIMA (Definição dos termos AR e MA)
model = pf.ARIMA(data=data, ar=11, ma=11, integ=0, target='transaction_amount')
model_fit = model.fit()

# Visualização do ajuste do modelo sobre a série temporal
model.plot_fit(figsize=(20,8))
plt.show() 
```
---
### 🔹 Abordagem 2: Detecção de Outliers com Isolation Forest 

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# Carga dos dados transacionais
data = pd.read_csv('transaction_data.csv')

# Inicialização do modelo definindo uma taxa de contaminação estimada de 1%
model = IsolationForest(contamination=0.01, random_state=42)

# Treinamento focado no montante das transações
model.fit(data[['transaction_amount']])

# Predição: (-1 para anomalias, 1 para transações normais)
data['anomaly'] = model.predict(data[['transaction_amount']])

# Filtragem de transações identificadas como suspeitas
anomalies = data[data['anomaly'] == -1]
print(f"Total de anomalias detectadas: {len(anomalies)}")
```

### 📈 Próximos Passos & Melhorias Futuras 

- Implementar abordagens híbridas (utilizar o resíduo do ARIMA como entrada para o Isolation Forest).
- Adicionar engenharia de atributos (Feature Engineering) baseada no comportamento do usuário (ex: média de gastos nas últimas 24h).
- Criar uma esteira de deploy simulada usando FastAPI para escoragem em tempo real.

---

### 👩‍💻 Autora
Feito com 💛 por Bruna Guimarães
