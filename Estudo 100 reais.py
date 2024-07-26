



############ Estudo Aportes de 100 Reais ########################### 


# Importa as bibliotecas

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from bcb import sgs
import statsmodels.api as sm
import yfinance as yf
import warnings
import datetime
warnings.filterwarnings("ignore")

# Escolhendo o período
inicio = "2020-01-01"
fim = datetime.date.today().strftime("%Y-%m-%d")

# Lembrando que o IMAB só começou em 2019-05-20

# Segregando ano
ano_inicio = datetime.datetime.strptime(inicio, "%Y-%m-%d").year
ano_fim = datetime.datetime.strptime(fim, "%Y-%m-%d").year

# Código CDI no BC
CDI = sgs.get({'CDI':4389})
# Ou CDI = sgs.get(('CDI', 4389), start='2019-12-29') # ou last = 12)
# dfs = sgs.get({'SELIC': 1178, 'CDI': 4389}, start_date = '1986-07-04')

# Convertendo em Taxa diária
CDI['IDI'] = np.cumprod((1 + CDI['CDI'] / 100) ** (1 / 252))
CDI['ret'] = CDI['IDI'].pct_change().fillna(0)

# Escolhendo a janela de período
cdi_janela = CDI.IDI.loc[str(ano_inicio):].pct_change().cumsum()
#cdi_janela = CDI.IDI.loc['2017':'2023'].pct_change().cumsum()

# gráfico
cdi_janela.plot(figsize=(12, 6))
plt.title('Acumulated CDI Returns', fontsize=10)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()

# Mudando para CDI Mensal
# Escolhendo a janela de período

CDI_mensal = CDI['IDI'].pct_change().resample('M').sum()
CDI_mensal = CDI_mensal.loc[str(ano_inicio):]

# Acumulado
# CDI_mensal = CDI_mensal.loc['2019':].cumsum()

CDI_mensal.plot(kind = 'bar', figsize=(12, 6))
plt.title('CDI Monthly Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()

CDI_mensal.head()
CDI_mensal.tail()

# Outros ativos

# Define the tickers for IMAB and IBOV
tickers = ['BOVA11.SA', 'IMAB11.SA']

# Download the historical data for the tickers
data = yf.download(tickers, start= inicio, end= fim)["Adj Close"]

# Select the adjusted close prices for IMAB and CDI
imab_prices = data['IMAB11.SA']
ibov_prices = data['BOVA11.SA']

# Calculate the monthly returns for IMAB and CDI
imab_returns = imab_prices.pct_change().resample('M').sum()
ibov_returns = ibov_prices.pct_change().resample('M').sum()

# juntando as bases
merged_data = pd.concat([imab_returns, ibov_returns, CDI_mensal], axis=1)
merged_data.columns = ['IMAB11', 'BOVA11', 'CDI']
merged_data = merged_data.dropna()
merged_data.head()
merged_data.tail()


######################################################

# Análise dos Dados - Backtest aporte Mensal

# Define the monthly investment amount
investment_amount = 100

# Calculate the number of months
num_months = len(merged_data)

# Initialize the investment values
ibovespa_investment = [investment_amount]
imab_investment = [investment_amount]
cdi_investment = [investment_amount]

# Calculate the investment values for each month
for i in range(1, num_months):
    ibovespa_investment.append(ibovespa_investment[i-1] * (1 + merged_data['BOVA11'].iloc[i]))
    imab_investment.append(imab_investment[i-1] * (1 + merged_data['IMAB11'].iloc[i]))
    cdi_investment.append(cdi_investment[i-1] * (1 + merged_data['CDI'].iloc[i]))

#Invertendo os retornos - ganho de cada um dos aportes
ibovespa_investment = ibovespa_investment[::-1]
imab_investment = imab_investment[::-1]
cdi_investment = cdi_investment[::-1]

# Create a DataFrame to store the investment values
investment_data = pd.DataFrame({
    'Date': merged_data.index,
    'Ibovespa': ibovespa_investment,
    'IMAB': imab_investment,
    'CDI': cdi_investment
})

# Create a second dataframe with cumulative investment values
cumulative_investment_data = investment_data.copy()
cumulative_investment_data['Ibovespa'] = cumulative_investment_data['Ibovespa'].cumsum()
cumulative_investment_data['IMAB'] = cumulative_investment_data['IMAB'].cumsum()
cumulative_investment_data['CDI'] = cumulative_investment_data['CDI'].cumsum()

# Convertendo para data
cumulative_investment_data['Date'] = pd.to_datetime(cumulative_investment_data['Date'])
cumulative_investment_data.set_index('Date', inplace=True)
cumulative_investment_data

# Print the cumulative investment data
print(cumulative_investment_data)

# Plot the investment values
cumulative_investment_data.plot.line()
plt.xlabel('Months')
plt.ylabel('Investment Value')
plt.title(f'Aporte de 100 reais Mensais desde {ano_inicio}')
plt.text(cumulative_investment_data.index[-1], cumulative_investment_data.iloc[-1].max(), f"{cumulative_investment_data.iloc[-1].max():.2f}", ha='right', va='bottom')

# Add the value of the last row for each column
for column in cumulative_investment_data.columns:
    plt.text(cumulative_investment_data.index[-1], cumulative_investment_data[column].iloc[-1], f"{cumulative_investment_data[column].iloc[-1]:.2f}", ha='right', va='bottom')

plt.show()



######################################################











