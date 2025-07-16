"""
1.freqtrader
integrates easily with major exchanges like Binance, Bybit, KuCoin, etc.
"""
# 1.1 Install
# git clone https://github.com/freqtrade/freqtrade.git
# cd freqtrade
# python -m venv .env
# .env\Scripts\activate
# pip install -r requirements.txt

# 1.2 Configure
# freqtrade new-config
# freqtrade new-strategy --strategy MyBinanceStrategy
# (will produce a .py document)

# 1.3 Search (fetch data)
# freqtrade download-data --exchange binance --quote-currency USDT --base-currency BTC --timeframe 5m

# 1.4 open the .py doc produced in 1.2, and edit (This is SMA):
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

class MyBinanceStrategy(IStrategy):
    timeframe = '5m'
    minimal_roi = {"0": 0.01}
    stoploss = -0.05
    trailing_stop = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma'] = dataframe['close'].rolling(20).mean()
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['sma']),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['sma']),
            'sell'
        ] = 1
        return dataframe

# 1.5 Backtest
# freqtrade backtesting --strategy MyBinanceStrategy





"""
2.backtrader
Medium-frequency strategies (e.g., minute/hour/day)
well-suited for use inside Jupyter Notebooks.
"""
import backtrader as bt

# 2.1. Define strategy
class SMACrossover(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(period=20)

    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.close()

# 2.2. Load your data (as a Pandas DataFrame)
data = bt.feeds.PandasData(dataname=your_dataframe)

# 2.3. Run the backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(SMACrossover)
cerebro.adddata(data)
cerebro.run()
cerebro.plot()
