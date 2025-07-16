There are many established modules that can fulfill this task. I tried these two: 
- freqtrade (not native to jupyter but it connects well with crypto world)
- backtrader (native to jupyter)

# 1.freqtrade
## 1.1 Install
I executed these 4 command lines in cmd (windows environment) one by one. 

```
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
python -m venv .env
.env\Scripts\activate
```

So far so good. But for the 5th line `pip install -r requirements.txt` , it cases issues...

<img width="866" height="120" alt="image" src="https://github.com/user-attachments/assets/dfcf1401-3d26-4923-94a8-1cff5e40f9c1" />

### *Troubleshooting*
The installation of ta-lib via pip often fails on Windows because it requires native C/C++ compilation, which depends on external build tools like Microsoft Visual C++ Build Tools. 

To work around this, we need to 
* manually download the precompiled .whl binary from a trusted source (e.g. [Gohlke’s repository](https://github.com/cgohlke/talib-build/releases/tag/v0.6.4) ) . Then, place it at the same directory as requirements.txt and install it `pip install ta_lib-0.6.4-cp313-cp313-win_amd64.whl`
* remove (or comment out) `ta-lib` from requirements.txt to avoid build errors during dependency installation.

And then, after the above 2 steps, we can input the 5th line `pip install -r requirements.txt`

## 1.2 Configure
```
python -m freqtrade new-config
python -m freqtrade new-strategy --strategy MyBinanceStrategy
```
It will jump out many options. for practice, we can just choose simple version.
Here is my choice:
<img width="932" height="301" alt="image" src="https://github.com/user-attachments/assets/48ae77eb-3489-4603-a5f9-6f112c1f8908" />

After this step, it will produce an `.py` in .\user_data\strategies
It is long and comprehensive. But for practice, we can replace it with a simple SMA strategy.

## 1.3 Search (fetch data)
```
python -m freqtrade download-data --exchange binance --pairs BTC/USDT --timeframes 5m
```
*Freqtrade has strict format requirements.*

## 1.4 Execute Self-defined Strategy
(kindly find the simple SMA strategy in the code repository)

## 1.5 Backtest
```
python -m freqtrade backtesting --strategy MyBinanceStrategy
```
*Freqtrade has very strict formatting requirements. I encounter "Configuration error" repeatedly. I revise the configuration to make it work — it runs for now, but I’m not sure how it will be in the future.*

Here is my configuration doc:
Refer to [freqtrade conj support pages](https://www.freqtrade.io/en/stable/configuration/)
```
{
  "dry_run": true,
  "strategy": "SampleStrategy",
  "exchange": {
    "name": "binance",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": ["BTC/USDT"]
  },
  "timeframe": "5m",
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "max_open_trades": 3,
  "tradable_balance_ratio": 0.99,
  "dry_run_wallet": 1000,
  "pairlists": [
    {
      "method": "StaticPairList"
    }
  ],
  "entry_pricing": {
    "price_side": "same",
    "price_last_balance": 0.5,
    "use_order_book": true,
    "order_book_top": 1,
    "check_depth_of_market": {
      "enabled": false,
      "bids_to_ask_delta": 1
    }
  },
  "exit_pricing": {
    "price_side": "same",
    "price_last_balance": 0.5,
    "use_order_book": true,
    "order_book_top": 1
  }
}

```

# 2. backtrader
It is native to jupyter environment. Much more convenient to use.

To install:
```
pip install backtrader
```



# 2. 




