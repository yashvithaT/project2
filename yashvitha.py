import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TRADING_DAYS = 252

#5 years of data
start_date = "2020-01-01"
end_date = "2025-01-01"
TRADING_DAYS = 252

tickers = [
    "NVDA",   # AI
    "MSFT",   # software
    "TSLA",   # transportation
    "AMZN",   # E-Commerce
    "V",      # payments
    "ABNB",   # travel
    "DIS",    # entertainment
    "XOM",    # energy
    "COST",   # manufacturing
    "FDX"     # logistics
]

adjusted_close_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

print("First 5 rows of adjusted close prices:")
print(adjusted_close_prices.head())

print("\nLast 5 rows of adjusted close prices:")
print(adjusted_close_prices.tail())

print("\nMissing values in adjusted close prices:")
print(adjusted_close_prices.isnull().sum())
#remove missing values
adjusted_close_prices = adjusted_close_prices.dropna()
#calculate daily returns
returns = adjusted_close_prices.pct_change().dropna()

print("\nDaily returns:")
print(returns.head())
#calculate average yearly return and volatility
annualized_returns = returns.mean() * TRADING_DAYS
annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS)

summary_stats = pd.DataFrame({
    "Annualized Return": annualized_returns * 100,
    "Annualized Volatility": annualized_volatility * 100
})
summary_stats = summary_stats.round(2)

summary_stats = summary_stats.sort_values(by="Annualized Return", ascending=False)
print("\nSummary statistics (sorted by Annualized Return):")
print(summary_stats)
#plot annualized returns
plt.figure(figsize=(10, 6))
summary_stats["Annualized Return"].plot(kind="bar", color="skyblue")
plt.title("Annualized Returns of Selected Stocks")
plt.xlabel("Stocks")
plt.ylabel("Annualized Return (%)")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

#plot cumulative returns
cumulative_returns = (1 + returns).cumprod()
plt.figure(figsize=(12, 6))

for stock in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)
plt.title("Cumulative Returns of Selected Stocks from 2020 to 2025")
plt.xlabel("Date")
plt.ylabel("Growth of $1 Investment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#plot correlation heatmap
correlation_matrix = returns.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

print("\nHighest Annualized Return:")
print(summary_stats.loc[summary_stats["Annualized Return"].idxmax()])
print("\nLowest Annualized Return:")
print(summary_stats.loc[summary_stats["Annualized Return"].idxmin()])
print("\nMost Volatile Stock:")
print(summary_stats.loc[summary_stats["Annualized Volatility"].idxmax()])
print("\nLeast Volatile Stock:")
print(summary_stats.loc[summary_stats["Annualized Volatility"].idxmin()])
