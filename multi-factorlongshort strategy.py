#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from IPython.display import display

class TechnicalIndicator:
    #Base class for calculating technical indicators.#
    def __init__(self, data):
        #Initialize with data.#
        self.data = data
    
    def calculate(self):
        #Calculate the indicator.#
        raise NotImplementedError("Base class method")

class EMAIndicator(TechnicalIndicator):
   #Calculates Exponential Moving Average (EMA)#
    def __init__(self, data, period):
       #Initialize with data and period.#
        super().__init__(data)
        self.period = period
    
    def calculate(self):
        #Calculate EMA.#
        return pd.Series(self.data).ewm(span=self.period, adjust=False).mean().values

class RSIIndicator(TechnicalIndicator):
    #Calculates Relative Strength Index (RSI).#
    def __init__(self, prices, period=14):
        #Initialize RSI with prices and period.#
        super().__init__(prices)
        self.period = period
    
    def calculate(self):
        #Calculate RSI.#
        prices = np.asarray(self.data, dtype=np.float64)
        if len(prices) < self.period + 1:
            return np.zeros(len(prices))
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gains = np.mean(gains[:self.period])
        avg_losses = np.mean(losses[:self.period])
        
        rsi = np.zeros(len(prices))
        
        if avg_losses < 1e-8:
            rsi[self.period] = 100.0 if avg_gains > 1e-8 else 50.0
        else:
            rs = avg_gains / avg_losses
            rsi[self.period] = 100 - (100 / (1 + rs))
        
        #Apply Wilder Smoothing#
        for i in range(self.period + 1, len(prices)):
            avg_gains = (avg_gains * (self.period - 1) + gains[i-1]) / self.period
            avg_losses = (avg_losses * (self.period - 1) + losses[i-1]) / self.period
            
            if avg_losses < 1e-10:
                rsi[i] = 100.0 if avg_gains > 1e-10 else 50.0
            else:
                rs = avg_gains / avg_losses
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi

class MACDIndicator(TechnicalIndicator):
    #Calculates Moving Average Convergence Divergence (MACD).#
    def __init__(self, prices, fast=12, slow=26, signal=9):
       #Initialize MACD with prices and parameters.#
        super().__init__(prices)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self):
       #Calculate MACD, signal line, and histogram.#
        fast_ema = EMAIndicator(self.data, self.fast).calculate()
        slow_ema = EMAIndicator(self.data, self.slow).calculate()
        macd = fast_ema - slow_ema
        signal_line = EMAIndicator(macd, self.signal).calculate()
        hist = macd - signal_line
        return macd, signal_line, hist

class CorrelationMatrix:
    #Calculates correlation matrix for given factors.#
    def __init__(self, factors):
        #Initialize with factors.#
        self.factors = np.array(factors)
    
    def calculate(self):
        #Calculate correlation matrix.#
        n = self.factors.shape[0]
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x = self.factors[i]
                y = self.factors[j]
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                cov = np.mean((x - mean_x) * (y - mean_y))
                std_x = np.std(x)
                std_y = np.std(y)
                matrix[i, j] = cov / (std_x * std_y) if std_x != 0 and std_y != 0 else 0
        return matrix

class BetaCalculator:
    #Calculates beta between token and BTC returns.#
    def __init__(self, token_returns, btc_returns):
        #Initialize with token and BTC returns.#
        self.token_returns = token_returns
        self.btc_returns = btc_returns
    
    def calculate(self):
        #Calculate beta.#
        n = min(len(self.token_returns), len(self.btc_returns))
        if n < 2:
            return 0.0
        token_returns = np.array(self.token_returns[:n])
        btc_returns = np.array(self.btc_returns[:n])
        mean_token = np.mean(token_returns)
        mean_btc = np.mean(btc_returns)
        cov = np.mean((token_returns - mean_token) * (btc_returns - mean_btc))
        var_btc = np.var(btc_returns)
        return cov / var_btc if var_btc != 0 else 0.0

class MarketCapClassifier:
    #Classifies market cap into categories.#
    @staticmethod
    def classify(market_cap):
        #Classify market cap.#
        if market_cap < 100_000_000:
            return 'Small'
        elif market_cap < 1_000_000_000:
            return 'Mid'
        elif market_cap < 10_000_000_000:
            return 'Large'
        return 'ExtraLarge'

class DataLoader:
    #Loads and processes JSON data from directory.#
    def __init__(self, directory):
        #Initialize with directory path.#
        self.directory = Path(directory)
        self.tokens = {}
    
    def load(self):
       #Load JSON data from directory.#
        if not self.directory.exists():
            print(f"Directory does not exist: {self.directory}")
            return self.tokens
        print(f"Scanning directory: {self.directory}")
        for file_path in self.directory.glob('*.json'):
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict) or 'date' not in data or 'data' not in data:
                    print(f"File {file_path} is missing 'date' or 'data' field")
                    continue
                timestamp = data['date']
                if not isinstance(timestamp, (int, float)) or timestamp < 1000000000:
                    print(f"File {file_path} has invalid timestamp: {timestamp}")
                    continue
                print(f"File {file_path} timestamp: {timestamp}, converted: {datetime.fromtimestamp(timestamp)}")
                for item in data['data']:
                    required_fields = ['symbol', 'price', 'volume24h', 'open_interest', 'funding',
                                      'market_cap', 'oi_volume24h', 'long_short_ratio', 'realized_vol']
                    if not all(field in item for field in required_fields) or '30' not in item['realized_vol']:
                        print(f"File {file_path} is missing required fields")
                        continue
                    symbol = item['symbol']
                    ohlc = {
                        'price': item['price'],
                        'volume': item['volume24h'],
                        'open_interest': item['open_interest'],
                        'funding_rate': item['funding'],
                        'market_cap': item['market_cap'],
                        'oi_volume_ratio': item['oi_volume24h'],
                        'long_short_ratio': item['long_short_ratio'],
                        'realized_vol': item['realized_vol']['30'],
                        'timestamp': timestamp
                    }
                    if symbol not in self.tokens:
                        self.tokens[symbol] = []
                    self.tokens[symbol].append(ohlc)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {file_path}: {e}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        for symbol in self.tokens:
            self.tokens[symbol] = sorted(self.tokens[symbol], key=lambda x: x['timestamp'])
        return self.tokens
    
    def read_signal_logs(self):
        #Read trading signal logs from file.#
        try:
            with open('trading_signals.log', 'r', encoding='utf-8') as f:
                logs = f.readlines()
                for line in logs:
                    if "Symbol" in line:
                        print(line.strip())
        except FileNotFoundError:
            print("No signal logs found.")
        except Exception as e:
            print(f"Error reading signal logs: {e}")

class BacktestEngine:
    #Executes backtesting for trading strategy.#
    def __init__(self, tokens, btc, split_timestamp, initial_capital, transaction_cost=0.001, slippage=0.0005):
        #Initialize backtest engine.#
        self.tokens = tokens
        self.btc = btc
        self.split_timestamp = split_timestamp
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.result = {
            'total_return': 0.0,
            'alpha_return': 0.0,
            'sharpe_ratio': 0.0,
            'trades': [],
            'historical_positions': [],
            'proposed_signals': {'signals': []},
            'correlation_matrices': {}
        }
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            # File handler
            file_handler = logging.FileHandler('trading_signals.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def calculate_btc_returns(self):
        #Calculate daily returns for BTC.#
        btc_returns = [0.0] * len(self.btc)
        for i in range(1, len(self.btc)):
            btc_returns[i] = self.btc[i]['price'] / self.btc[i - 1]['price'] - 1.0
        return btc_returns
    
    def prepare_training_data(self):
        #Prepare training data for the model.#
        train_features = []
        train_labels = []
        for symbol, data in self.tokens.items():
            if len(data) < 14 + 5:
                continue
            prices = [d['price'] for d in data]
            rsi = RSIIndicator(prices).calculate()
            macd, _, _ = MACDIndicator(prices).calculate()
            for i in range(14, len(data) - 5):
                if data[i]['timestamp'] >= self.split_timestamp:
                    continue
                features = [
                    rsi[i],
                    macd[i],
                    data[i]['oi_volume_ratio'],
                    data[i]['funding_rate'],
                    data[i]['market_cap'],
                    data[i]['long_short_ratio'],
                    data[i]['realized_vol']
                ]
                target = 1.0 if prices[i + 5] / prices[i] - 1 > 0.05 else 0.0
                train_features.append(features)
                train_labels.append(target)
        return train_features, train_labels
    
    def train_model(self, train_features, train_labels):
        #Train XGBoost model.#
        model = None
        if train_features:
            dtrain = xgb.DMatrix(train_features, label=train_labels)
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'eta': 0.3,
                'eval_metric': 'logloss'
            }
            model = xgb.train(params, dtrain, num_boost_round=100)
        return model
    
    def simulate_trading(self, model):
       #Simulate trading based on model predictions.# 
        active_trades = {}
        daily_returns = []
        enter_threshold = 0.6     # modify here to change the long and short boundary 
        exit_threshold = 0.4
        take_profit_perc = 0.01
        position_size = self.initial_capital / 10.0

        for symbol, data in self.tokens.items():
            if len(data) < 14:
                continue
            prices = [d['price'] for d in data]
            oi_volume_ratios = [d['oi_volume_ratio'] for d in data]
            funding_rates = [d['funding_rate'] for d in data]
            long_short_ratios = [d['long_short_ratio'] for d in data]
            realized_vols = [d['realized_vol'] for d in data]
            token_returns = [0.0] * len(data)
            for i in range(1, len(data)):
                token_returns[i] = prices[i] / prices[i - 1] - 1.0

            rsi = RSIIndicator(prices).calculate()
            macd, _, _ = MACDIndicator(prices).calculate()

            if len(rsi) >= 14:
                factors = [rsi, macd, oi_volume_ratios, funding_rates, long_short_ratios, realized_vols]
                self.result['correlation_matrices'][symbol] = CorrelationMatrix(factors).calculate()

            for i in range(14, len(data)):
                if data[i]['timestamp'] < self.split_timestamp:
                    continue

                if symbol in active_trades:
                    self.result['historical_positions'].append({
                        'symbol': symbol,
                        'entry_price': active_trades[symbol]['entry_price'],
                        'current_price': data[i]['price'],
                        'direction': active_trades[symbol]['direction'],
                        'entry_time': datetime.fromtimestamp(data[i]['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                        'position_size': position_size
                    })

                features = [
                    rsi[i],
                    macd[i],
                    data[i]['oi_volume_ratio'],
                    data[i]['funding_rate'],
                    data[i]['market_cap'],
                    data[i]['long_short_ratio'],
                    data[i]['realized_vol']
                ]
                pred = 0.5 if not model else model.predict(xgb.DMatrix([features]))[0]

                cap_category = MarketCapClassifier.classify(data[i]['market_cap'])

                long_condition = (cap_category in ['Small', 'Mid']) and \
                                 data[i]['oi_volume_ratio'] > 0.05 and \
                                 data[i]['funding_rate'] > 0.0 and \
                                 pred > enter_threshold
                short_condition = cap_category == 'ExtraLarge' and \
                                  data[i]['oi_volume_ratio'] < 0.05 and \
                                  data[i]['funding_rate'] < 0.0 and \
                                  pred < exit_threshold

                if symbol not in active_trades and long_condition:
                    active_trades[symbol] = {
                        'symbol': symbol,
                        'direction': 'Long',
                        'entry_price': data[i]['price'],
                        'entry_time': data[i]['timestamp'],
                        'active': True
                    }
                    print(f"Enter Long {symbol} at price: {data[i]['price']} time: {datetime.fromtimestamp(data[i]['timestamp'])}")
                elif symbol not in active_trades and short_condition:
                    active_trades[symbol] = {
                        'symbol': symbol,
                        'direction': 'Short',
                        'entry_price': data[i]['price'],
                        'entry_time': data[i]['timestamp'],
                        'active': True
                    }
                    print(f"Enter Short {symbol} at price: {data[i]['price']} time: {datetime.fromtimestamp(data[i]['timestamp'])}")
                elif symbol in active_trades:
                    trade = active_trades[symbol]
                    price_change = (data[i]['price'] - trade['entry_price']) / trade['entry_price'] if trade['direction'] == 'Long' else \
                                   (trade['entry_price'] - data[i]['price']) / trade['entry_price']
                    if price_change >= take_profit_perc or pred < exit_threshold:
                        trade['exit_price'] = data[i]['price']
                        trade['exit_time'] = data[i]['timestamp']
                        trade['return_perc'] = price_change
                        trade['active'] = False
                        self.result['trades'].append(trade)
                        daily_returns.append(price_change)
                        self.result['total_return'] += price_change
                        print(f"Exit {trade['direction']} {symbol} at price: {trade['exit_price']} time: {datetime.fromtimestamp(trade['exit_time'])} return: {price_change * 100:.2f}%")
                        del active_trades[symbol]

        return daily_returns
    
    def calculate_metrics(self, daily_returns, btc_returns):
      #Calculate performance metrics.#
        beta = BetaCalculator(daily_returns, btc_returns).calculate()
        btc_total_return = sum(btc_returns)
        self.result['alpha_return'] = self.result['total_return'] - beta * btc_total_return

        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_dev = np.std(daily_returns)
            self.result['sharpe_ratio'] = mean_return / std_dev * np.sqrt(252) if std_dev != 0 else 0.0
    
    def generate_signals(self, model):
       #Generate proposed trading signals.#
        for symbol, data in self.tokens.items():
            if not data or data[-1]['timestamp'] < self.split_timestamp:
                continue
            prices = [d['price'] for d in data]
            rsi = RSIIndicator(prices).calculate()
            macd, signal, hist = MACDIndicator(prices).calculate()
            i = len(prices) - 1
            features = [
                rsi[i],
                macd[i],
                data[i]['oi_volume_ratio'],
                data[i]['funding_rate'],
                data[i]['market_cap'],
                data[i]['long_short_ratio'],
                data[i]['realized_vol']
            ]
            pred = 0.5 if not model else model.predict(xgb.DMatrix([features]))[0]
            self.logger.info(f"Symbol: {symbol}, pred: {pred:.5f}")
            cap_category = MarketCapClassifier.classify(data[i]['market_cap'])
            if pred > 0.5:
                self.result['proposed_signals']['signals'].append({
                    'symbol': symbol,
                    'direction': 'Long',
                    'price': data[i]['price'],
                    'timestamp': datetime.fromtimestamp(data[i]['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'confidence': float(pred)
                })

    def run(self):
        #Run the backtest.#
        btc_returns = self.calculate_btc_returns()
        train_features, train_labels = self.prepare_training_data()
        model = self.train_model(train_features, train_labels)
        daily_returns = self.simulate_trading(model)
        self.calculate_metrics(daily_returns, btc_returns)
        self.generate_signals(model)
        return self.result

def main():
    "Main function to execute the backtest."
    directory = "D:\\assignment\\data"
    loader = DataLoader(directory)
    tokens = loader.load()

    if not tokens:
        print("No token data loaded. Exiting program.")
        return
    btc = next((data_list for symbol, data_list in tokens.items() if 'BTC' in symbol), None)
    if not btc:
        print("BTC data not found!")
        return
    timestamps = [d['timestamp'] for data in tokens.values() for d in data]
    if not timestamps:
        print("No valid timestamp data found!")
        return
# here is the place that you can modify the ratio of train data and testing date, for example, if you change 0.8 to 0.5, it mean that 
# you will have more testing data, less train data
    split_timestamp = min(timestamps) + (max(timestamps) - min(timestamps)) * 0.8
    initial_capital = 100000.0
    engine = BacktestEngine(
        tokens=tokens,
        btc=btc,
        split_timestamp=split_timestamp,
        initial_capital=initial_capital,
        transaction_cost=0.001,
        slippage=0.0005
    )
    result = engine.run()

    print("\n=== Backtest Results ===")
    print(f"Total Return: {result['total_return'] * 100:.2f}%")
    print(f"Alpha Return (BTC-adjusted): {result['alpha_return'] * 100:.2f}%")
    print(f"Sharpe Ratio (Annualized): {result['sharpe_ratio']:.2f}")

    print("\n=== Trade Records ===")
    trades_df = pd.DataFrame(result['trades'])
    if not trades_df.empty:
        trades_df['entry_time'] = trades_df['entry_time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        trades_df['exit_time'] = trades_df['exit_time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        trades_df['return_perc'] = trades_df['return_perc'] * 100
        display(trades_df[['symbol', 'direction', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'return_perc']])

    print("\n=== Historical Positions ===")
    positions_df = pd.DataFrame(result['historical_positions'])
    if not positions_df.empty:
        display(positions_df[['symbol', 'direction', 'entry_time', 'entry_price', 'current_price', 'position_size']])

    print("\n=== Correlation Matrices ===")
    for symbol, matrix in result['correlation_matrices'].items():
        print(f"Correlation Matrix for {symbol}:")
        factors = ['RSI', 'MACD', 'OI/Vol', 'Funding', 'LS Ratio', 'Real Vol']
        matrix_df = pd.DataFrame(matrix, index=factors, columns=factors)
        display(matrix_df.round(2))
# I think we only need to  output the final day's trading signal, so I just track the final day
    print("\n=== Proposed Trading Signals ===")
    signals_df = pd.DataFrame(result['proposed_signals']['signals'])
    if not signals_df.empty:
        display(signals_df)

if __name__ == "__main__":
    main()


# In[ ]:




