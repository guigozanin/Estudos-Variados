
# Estudo Estudo Correlação BTC

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 


# lets import more data
df = yf.download(['BTC-USD', # bitcoin
                  'GC=F', # ouro 
                  '^IXIC']) # nasdaq

df.tail()


# Closing price
df = df['Adj Close'].dropna()
df = df.dropna()

df.tail()
df.columns = ['Bitcoin', 'Ouro', 'Nasdaq']



# Log of percentage change
returns = df.pct_change().apply(lambda x: np.log(1+x)).dropna()
returns



# Janelas móveis separadas - ESCOLHA O PERIODO
bitcoin_ouro = returns['Bitcoin'].rolling(window=252).corr(returns['Ouro'])
bitcoin_nasdaq = returns['Bitcoin'].rolling(window=21).corr(returns['Nasdaq'])

# Junta as bases
rolling_returns = pd.concat([bitcoin_ouro, bitcoin_nasdaq], axis=1)
rolling_returns.columns = ["Bitcoin x Ouro","Bitcoin x Nasdaq" ]


# Gráfico
# Plotting the rolling returns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Adicionando um título geral
fig.suptitle('Correlação Bitcoin, Janelas Móveis de 252 dias')


# Plotting the correlation between Bitcoin and Ouro
ax1.plot(rolling_returns.index, rolling_returns["Bitcoin x Ouro"], color='darkgoldenrod')
ax1.set_title("Bitcoin x Ouro")
ax1.set_xlabel("Date")
ax1.set_ylabel("Correlation")
ax1.set_ylim([-1, 1])  # Set y-axis limits


# Plotting the correlation between Bitcoin and Nasdaq
ax2.plot(rolling_returns.index, rolling_returns["Bitcoin x Nasdaq"], color='darkblue')
ax2.set_title("Bitcoin x Nasdaq")
ax2.set_xlabel("Date")
ax2.set_ylabel("Correlation")
ax2.set_ylim([-1, 1])  # Set y-axis limits

# Adicionando a legenda
plt.figtext(0.5, 0.01, 'Elaboração @Gui_Zanin - Base de dados: Yahoo Finance', ha='center', va='center')

# Displaying the plots
plt.tight_layout()
plt.show()
