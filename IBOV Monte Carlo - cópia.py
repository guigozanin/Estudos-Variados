





from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

import yfinance as yf




############
# Variáveis que brincamos

# Número de simulações desejadas
num_simulacoes = 100
num_ativos_cart = 5
ano_inicio = 2023
ano_fim = 2023
taxa_anual = 0.02 # adm


############






# Ibov

HOJE = datetime.today()
end = HOJE
start = end - timedelta(days=365*26)
df_ibov = yf.download("^BVSP", start=start, end=end)["Adj Close"]
df_ibov_pct = df_ibov.pct_change()


#Carrega composição histórica quadrimestral do IBRX

# Dados obtidos do Museu da B3 e da Com Dinheiro
# Muito importante para tirar o viés do sobrevivente
# A estratégia irá olhar para composição do quadrimestre para montar a carteira do período.


column_comp_indices_names = ['quad', 'indice', 'ticker']

df_comp_indices = pd.read_csv("/Users/guilhermerenatorosslerzanin/Antigo/Python/bases/composicao_indice_ibrx.csv", sep=";")
df_comp_indices.columns = columns=column_comp_indices_names


def obter_quad(data):
    if data.month in (1, 2, 3, 4):
        return str(1)+"Q"+str(data.year)
    if data.month in (5, 6, 7, 8):
        return str(2)+"Q"+str(data.year)
    if data.month in (9, 10, 11, 12):
        return str(3)+"Q"+str(data.year)
    
#Carrega os arquivos de cotação e cria vários data frames:

#df_adj_close_daily: fechamento diário;
#df_daily_returns: retorno diário;
#df_adj_close_monthly: fechamento mensal;
#df_monthly_returns: retorno mensal;


def ler_arquivo_cotacoes():
    concat_df = pd.read_csv("/Users/guilhermerenatorosslerzanin/Antigo/Python/bases/all_quotes.csv", parse_dates=["data"])
    return concat_df


df_adj_close_daily = ler_arquivo_cotacoes()
df_adj_close_daily = df_adj_close_daily.set_index("data")
df_adj_close_daily = df_adj_close_daily.sort_index(ascending=True)


df_daily_returns = df_adj_close_daily.pct_change()


df_adj_close_monthly = df_adj_close_daily.resample("M").last()

df_monthly_returns = df_adj_close_monthly.pct_change()


df_monthly_melted = pd.melt(df_adj_close_monthly.reset_index(), id_vars=['data'], var_name='ticker', value_name='preco_fechamento_ajustado')
df_monthly_melted["quad"] = df_monthly_melted["data"].apply(obter_quad)
df_monthly_melted["ano"] = df_monthly_melted["data"].dt.year
df_monthly_melted["mes"] = df_monthly_melted["data"].dt.month


# Base diária

df_daily_melted = pd.melt(df_daily_returns.reset_index(), id_vars=['data'], var_name='ticker', value_name='variacao_fechamento')
df_daily_melted["quad"] = df_daily_melted["data"].apply(obter_quad)
df_daily_melted["ano"] = df_daily_melted["data"].dt.year
df_daily_melted["mes"] = df_daily_melted["data"].dt.month

df_daily_melted = df_daily_melted.fillna(0)

# Calculando para multiplicar
# df_daily_melted['variacao_fechamento'] = df_daily_melted['variacao_fechamento'] + 1


# Escolhendo 5 ativos aleatórios no índice, por trimestre
df_comp_indices

df_comp_indices['ticker_count'] = df_comp_indices.groupby('quad')['ticker'].transform('count')
df_comp_indices['ano'] = df_comp_indices['quad'].str[-4:]
# Converte a coluna 'quad' para inteiro para garantir a ordenação correta
df_comp_indices['ano'] = df_comp_indices['ano'].astype(int)

# Ordena o DataFrame pela coluna 'quad' em ordem crescente
df_comp_indices = df_comp_indices.sort_values(by='ano').reset_index(drop=True)
# Filtra o DataFrame para incluir apenas os anos após 2019
df_comp_indices = df_comp_indices[(df_comp_indices['ano'] >= ano_inicio) & (df_comp_indices['ano'] <= ano_fim)].reset_index(drop=True)






# DataFrame final para armazenar todas as simulações
df_carteira_final = pd.DataFrame()

for i in range(num_simulacoes):
    # Passo 6: Escolhendo 5 ativos aleatórios no índice, por trimestre
    selected_tickers = []
    for quad in df_comp_indices['quad'].unique():
        tickers = df_comp_indices[df_comp_indices['quad'] == quad]['ticker'].tolist()
        selected_tickers.extend(np.random.choice(tickers, size=5, replace=False)) # 5 ativos aleatórios

    selected_tickers_df = pd.DataFrame({
        'quad': np.repeat(df_comp_indices['quad'].unique(), 5),
        'ticker': selected_tickers
    })

    # Passo 7: Pegando o histórico dos ativos filtrados
    filtered_df = df_daily_melted.merge(selected_tickers_df, on=['quad', 'ticker'])
    filtered_df = filtered_df.sort_values(by='data')

    # Passo 8: Calculando o retorno da carteira
    df = filtered_df.copy()
    df['num_tickers'] = df.groupby('data')['ticker'].transform('count')
    df['peso'] = 1 / df['num_tickers']
    df['retorno_ponderado'] = df['peso'] * df['variacao_fechamento']
    
    # Calcula o retorno diário da carteira para a simulação atual
    df_carteira = df.groupby('data').agg(retorno_diario=('retorno_ponderado', 'sum')).reset_index()
    
    # Renomeia a coluna de retorno diário para incluir o número da simulação
    df_carteira.rename(columns={'retorno_diario': f'retorno_diario_simulacao_{i+1}'}, inplace=True)

    # Junta os retornos diários desta simulação ao DataFrame final
    if df_carteira_final.empty:
        df_carteira_final = df_carteira
    else:
        df_carteira_final = df_carteira_final.merge(df_carteira, on='data', how='left')

df_carteira_final




# DataFrame para armazenar os retornos acumulados
df_carteira_acumulado = pd.DataFrame()
df_carteira_acumulado['data'] = df_carteira_final['data']  # Copia a coluna 'data'

# Calcula o retorno acumulado para cada coluna de simulação e armazena no novo DataFrame
for col in df_carteira_final.columns[1:]:  # Ignora a primeira coluna, que é 'data'
    df_carteira_acumulado[f'{col}_acumulado'] = (1 + df_carteira_final[col]).cumprod() - 1

df_carteira_acumulado




# Comparando com o Ibov
df_carteira_mont = df_carteira_acumulado.merge(df_ibov_pct, left_on='data', right_index=True, how='left')
df_carteira_mont = df_carteira_mont.fillna(0)
df_carteira_mont.rename(columns={"Adj Close": "retorno_ibov"}, inplace=True)
df_carteira_mont['retorno_acumulado_ibov'] = ((df_carteira_mont['retorno_ibov']+1).cumprod())
df_carteira_mont['retorno_acumulado_ibov']  = df_carteira_mont['retorno_acumulado_ibov'] - 1
df_carteira_mont = df_carteira_mont.drop('retorno_ibov', axis=1)



# Gráfico

plt.plot(df_carteira_mont['data'], df_carteira_mont.iloc[:, 1:], color='silver')
plt.plot(df_carteira_mont['data'], df_carteira_mont['retorno_acumulado_ibov'], color='blue', label='IBOV')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.figtext(0.01, 0.01, 'Elaboração @Gui_Zanin - Base de dados: Yahoo Finance')
plt.title('Comparação do Retorno Acumulado Carteiras Aleatórias')
plt.legend()
#plt.savefig('ibov_aleatorio.png')
plt.show()





# Ativos Monte  Carlos e estatística



# Formatter to convert y-axis values to percentages
percent_formatter = FuncFormatter(lambda x, _: f'{x * 100:.0f}%')


# Select the last date (final time step) from your cumulative returns DataFrame
terminal_returns = df_carteira_mont.iloc[-1, 1:]  # Exclude the 'data' column
retorno_acumulado_ibov_terminal = df_carteira_mont['retorno_acumulado_ibov'].iloc[-1]  # Get the final IBRX return

# Set up the figure with a grid for the two plots
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15)

# Left plot: Time series of cumulative returns for all simulations and IBRX
ax0 = fig.add_subplot(gs[0])
for col in df_carteira_mont.columns[1:-1]:  # Plot all simulation columns except 'data' and 'retorno_acumulado_ibov'
    ax0.plot(df_carteira_mont['data'], df_carteira_mont[col], color='silver', alpha=0.7)
ax0.plot(df_carteira_mont['data'], df_carteira_mont['retorno_acumulado_ibov'], color='blue', label='IBOV')
ax0.set_xlabel('Período')
ax0.set_ylabel('Retorno Acumulado (%)')
ax0.set_title('Retornos Acumulados das Carteiras Aleatórias')
ax0.legend()
ax0.grid()

# Apply percentage formatting to y-axis
ax0.yaxis.set_major_formatter(percent_formatter)

# Right plot: Histogram of terminal returns
ax1 = fig.add_subplot(gs[1])
ax1.hist(terminal_returns, bins=30, color='silver', edgecolor='silver', orientation='horizontal')
ax1.axhline(y=retorno_acumulado_ibov_terminal, color='blue', linestyle='--', linewidth=2.5, label='Retorno Terminal IBOV')
ax1.set_xlabel('Probabilidade')
ax1.set_title('Distribuição de Retornos')
ax1.legend()
ax1.grid(axis='y')

# Sync y-axis limits to make both x-axis labels the same size
ymin, ymax = ax0.get_ylim()
ax1.set_ylim(ymin, ymax)

# Apply percentage formatting to y-axis on histogram as well
ax1.yaxis.set_major_formatter(percent_formatter)

# Final adjustments
plt.figtext(0.01, 0.01, 'Elaboração @Gui_Zanin - Base de dados: Economática')
#plt.tight_layout()
plt.show()





# Calculando estatatísticas


# Calcular a porcentagem de simulações que ficaram acima do retorno do Ibovespa
simulacoes_acima_ibov = (terminal_returns[terminal_returns.index.str.contains('simulacao')] > retorno_acumulado_ibov_terminal).sum()
total_simulacoes = terminal_returns[terminal_returns.index.str.contains('simulacao')].count()

percentagem_acima_ibov = (simulacoes_acima_ibov / total_simulacoes) * 100

# Filtrar apenas as simulações
simulacoes = terminal_returns[terminal_returns.index.str.contains('simulacao')]

# Calcular estatísticas
media = simulacoes.mean()
mediana = simulacoes.median()
desvio_padrao = simulacoes.std()
maior_retorno = simulacoes.max()
menor_retorno = simulacoes.min()

# Apresentar os resultados
print(f'Média dos retornos: {media * 100:.2f}%')
print(f'Média (mediana) dos retornos: {mediana * 100:.2f}%')
print(f'O retorno do IBOV no período foi de: {retorno_acumulado_ibov_terminal * 100:.2f}%')
print(f'Percentagem de simulações acima do retorno acumulado do IBOV: {percentagem_acima_ibov:.2f}%')
print(f'Desvio padrão dos retornos: {desvio_padrao * 100:.2f}%')
print(f'Maior retorno: {maior_retorno * 100:.2f}%')
print(f'Menor retorno: {menor_retorno * 100:.2f}%')




############

# Excluíndo taxa de adm

# Calcular a taxa diária equivalente
taxa_diaria = (1 + taxa_anual) ** (1 / 252) - 1

# Exibir a taxa diária
#print(f"Taxa diária: {taxa_diaria:.8f}")

# Base de retornos bruta
df_carteira_final

# Subtrair a taxa_diaria de todas as colunas numéricas
df_carteira_final_sem_adm = df_carteira_final.apply(lambda x: x - taxa_diaria if pd.api.types.is_numeric_dtype(x) else x)


# DataFrame para armazenar os retornos acumulados
df_carteira_acumulado_sem_adm = pd.DataFrame()
df_carteira_acumulado_sem_adm['data'] = df_carteira_final_sem_adm['data']  # Copia a coluna 'data'

# Calcula o retorno acumulado para cada coluna de simulação e armazena no novo DataFrame
for col in df_carteira_final_sem_adm.columns[1:]:  # Ignora a primeira coluna, que é 'data'
    df_carteira_acumulado_sem_adm[f'{col}_acumulado'] = (1 + df_carteira_final_sem_adm[col]).cumprod() - 1

df_carteira_acumulado_sem_adm




# Comparando com o Ibov
df_carteira_mont_sem_adm = df_carteira_acumulado_sem_adm.merge(df_ibov_pct, left_on='data', right_index=True, how='left')
df_carteira_mont_sem_adm = df_carteira_mont_sem_adm.fillna(0)
df_carteira_mont_sem_adm.rename(columns={"Adj Close": "retorno_ibov"}, inplace=True)
df_carteira_mont_sem_adm['retorno_acumulado_ibov'] = ((df_carteira_mont_sem_adm['retorno_ibov']+1).cumprod())
df_carteira_mont_sem_adm['retorno_acumulado_ibov']  = df_carteira_mont_sem_adm['retorno_acumulado_ibov'] - 1
df_carteira_mont_sem_adm = df_carteira_mont_sem_adm.drop('retorno_ibov', axis=1)



# Gráfico

plt.plot(df_carteira_mont_sem_adm['data'], df_carteira_mont_sem_adm.iloc[:, 1:], color='silver')
plt.plot(df_carteira_mont_sem_adm['data'], df_carteira_mont_sem_adm['retorno_acumulado_ibov'], color='red', label='IBOV')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.figtext(0.01, 0.01, 'Elaboração @Gui_Zanin - Base de dados: Yahoo Finance')
plt.title('Comparação do Retorno Acumulado Carteiras Aleatórias COM 2% ADM')
plt.legend()
#plt.savefig('ibov_aleatorio.png')
plt.show()





# Ativos Monte  Carlos e estatística



# Formatter to convert y-axis values to percentages
percent_formatter = FuncFormatter(lambda x, _: f'{x * 100:.0f}%')


# Select the last date (final time step) from your cumulative returns DataFrame
terminal_returns = df_carteira_mont_sem_adm.iloc[-1, 1:]  # Exclude the 'data' column
retorno_acumulado_ibov_terminal = df_carteira_mont_sem_adm['retorno_acumulado_ibov'].iloc[-1]  # Get the final IBOV return

# Set up the figure with a grid for the two plots
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15)

# Left plot: Time series of cumulative returns for all simulations and IBOV
ax0 = fig.add_subplot(gs[0])
for col in df_carteira_mont_sem_adm.columns[1:-1]:  # Plot all simulation columns except 'data' and 'retorno_acumulado_ibov'
    ax0.plot(df_carteira_mont_sem_adm['data'], df_carteira_mont_sem_adm[col], color='silver', alpha=0.7)
ax0.plot(df_carteira_mont_sem_adm['data'], df_carteira_mont_sem_adm['retorno_acumulado_ibov'], color='red', label='IBOV')
ax0.set_xlabel('Período')
ax0.set_ylabel('Retorno Acumulado (%)')
ax0.set_title('Retornos Acumulados das Carteiras Aleatórias COM 2% ADM')
ax0.legend()
ax0.grid()

# Apply percentage formatting to y-axis
ax0.yaxis.set_major_formatter(percent_formatter)

# Right plot: Histogram of terminal returns
ax1 = fig.add_subplot(gs[1])
ax1.hist(terminal_returns, bins=30, color='silver', edgecolor='silver', orientation='horizontal')
ax1.axhline(y=retorno_acumulado_ibov_terminal, color='red', linestyle='--', linewidth=2.5, label='Retorno Terminal IBOV')
ax1.set_xlabel('Probabilidade')
ax1.set_title('Distribuição de Retornos')
ax1.legend()
ax1.grid(axis='y')

# Sync y-axis limits to make both x-axis labels the same size
ymin, ymax = ax0.get_ylim()
ax1.set_ylim(ymin, ymax)

# Apply percentage formatting to y-axis on histogram as well
ax1.yaxis.set_major_formatter(percent_formatter)

# Final adjustments
plt.figtext(0.01, 0.01, 'Elaboração @Gui_Zanin - Base de dados: Economática')
#plt.tight_layout()
plt.show()





# Calculando estatatísticas


# Calcular a porcentagem de simulações que ficaram acima do retorno do Ibovespa
simulacoes_acima_ibov = (terminal_returns[terminal_returns.index.str.contains('simulacao')] > retorno_acumulado_ibov_terminal).sum()
total_simulacoes = terminal_returns[terminal_returns.index.str.contains('simulacao')].count()

percentagem_acima_ibov = (simulacoes_acima_ibov / total_simulacoes) * 100

# Filtrar apenas as simulações
simulacoes = terminal_returns[terminal_returns.index.str.contains('simulacao')]

# Calcular estatísticas
media = simulacoes.mean()
mediana = simulacoes.median()
desvio_padrao = simulacoes.std()
maior_retorno = simulacoes.max()
menor_retorno = simulacoes.min()

# Apresentar os resultados
print(f'Média dos retornos: {media * 100:.2f}%')
print(f'Média (mediana) dos retornos: {mediana * 100:.2f}%')
print(f'O retorno do IBOV no período f foi de: {retorno_acumulado_ibov_terminal * 100:.2f}%')
print(f'Percentagem de simulações acima do retorno acumulado do IBOV: {percentagem_acima_ibov:.2f}%')
print(f'Desvio padrão dos retornos: {desvio_padrao * 100:.2f}%')
print(f'Maior retorno: {maior_retorno * 100:.2f}%')
print(f'Menor retorno: {menor_retorno * 100:.2f}%')






############























