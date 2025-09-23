import matplotlib.pyplot as plt
import pandas as pd
import yfinance as finance

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)
SPY_dataset = finance.download('^GSPC', '2015-01-01')

print(SPY_dataset.head(10))
print('\n \n NAN\'s \n', SPY_dataset.isna().sum())

features = SPY_dataset[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
features.fillna(2000, inplace=True)
print('\nLength', len(features))

SPY_dataset.Close.plot()
plt.show()