# backend/model_core.py

import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import backtrader as bt
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- Backtrader Classes ---
class PandasRegimeFeed(bt.feeds.PandasData):
    lines = ('regime_code',)
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 0),
        ('high', 1),
        ('low', 2),
        ('close', 3),
        ('volume', 4),
        ('openinterest', 5),
        ('regime_code', 6)
    )

class RegimeStrategy(bt.Strategy):
    def __init__(self):
        self.regime_map = {0: 'Bull', 1: 'Bear', 2: 'Sideways', 3: 'Volatile'}
        self.order = None

    def next(self):
        if self.order:
            self.cancel(self.order)
        regime_code = int(self.datas[0].regime_code[0])
        regime = self.regime_map.get(regime_code, 'Unknown')
        if regime == 'Bull' and not self.position:
            self.order = self.buy(size=100)
        elif regime == 'Bear' and not self.position:
            self.order = self.sell(size=50)
        elif regime in ['Sideways', 'Volatile'] and self.position:
            self.order = self.close()

# --- Core Functions ---
def get_market_data(ticker, start_date="2015-01-01", end_date="2024-12-31"):
    try:
        # Download data with auto_adjust to fix the warning
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Fix MultiIndex columns issue
        if isinstance(data.columns, pd.MultiIndex):
            # For single ticker, flatten the MultiIndex by taking first level
            new_columns = []
            for col in data.columns:
                if isinstance(col, tuple):
                    # For tuples like ('Close', 'AAPL'), take the first part
                    new_columns.append(col[0])
                else:
                    new_columns.append(col)
            data.columns = new_columns
        
        # Remove any duplicate columns that might have been created
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Ensure we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
            
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Columns after processing: {data.columns.tolist()}")
        
        return data.dropna()
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        raise

def engineer_features(data):
    try:
        data = data.copy()  # Work with a copy
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['macd'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        data['rolling_mean'] = data['Close'].rolling(window=20).mean()
        data['rolling_std'] = data['Close'].rolling(window=20).std()
        data['bb_width'] = 2 * data['rolling_std'] / data['rolling_mean']
        data['atr'] = (data['High'] - data['Low']).rolling(14).mean()
        return data
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def train_hmm_model(data, features):
    try:
        feature_data = data[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_data)
        model = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_scaled)
        regimes = model.predict(X_scaled)
        return model, scaler, regimes, feature_data.index
    except Exception as e:
        logger.error(f"Error training HMM model: {str(e)}")
        raise

def label_regimes(data, regimes, feature_index):
    """Label regimes based on statistical properties"""
    try:
        # Ensure we're working with a copy of the relevant data
        regime_data = data.loc[feature_index].copy()
        
        # Add regime information
        regime_data['regime'] = regimes
        
        # Verify we have required columns
        if 'log_return' not in regime_data.columns:
            regime_data['log_return'] = np.log(regime_data['Close'] / regime_data['Close'].shift(1))
        
        # Calculate stats - handle potential NaN values
        stats = regime_data.groupby('regime')['log_return'].agg(['mean', 'std']).dropna()
        
        if stats.empty:
            raise ValueError("Could not calculate regime statistics - insufficient data")
            
        regime_labels = {}
        for idx, row in stats.iterrows():
            if row['mean'] < -0.001:
                regime_labels[idx] = 'Bear'
            elif row['mean'] > 0.002:
                regime_labels[idx] = 'Bull'
            elif row['std'] > 0.02:
                regime_labels[idx] = 'Volatile'
            else:
                regime_labels[idx] = 'Sideways'

        # Apply labels
        regime_data['regime_label'] = regime_data['regime'].map(regime_labels)
        regime_data['regime_code'] = regime_data['regime_label'].map({
            'Bull': 0, 'Bear': 1, 'Sideways': 2, 'Volatile': 3
        }).fillna(2)  # Default to Sideways if unknown
        
        return regime_data, regime_labels

    except Exception as e:
        logger.error(f"Error labeling regimes: {str(e)}")
        raise

def run_backtrader_simulation(data):
    try:
        # Create copy and ensure proper column names
        df_bt = data[['Open', 'High', 'Low', 'Close', 'Volume', 'regime_code']].copy()
        
        # Convert index to datetime if it isn't already
        if not isinstance(df_bt.index, pd.DatetimeIndex):
            df_bt.index = pd.to_datetime(df_bt.index)
            
        # Rename columns to Backtrader's expected format
        df_bt.columns = ['open', 'high', 'low', 'close', 'volume', 'regime_code']
        df_bt['openinterest'] = 0  # Required by Backtrader
        
        # Ensure all numeric data
        df_bt = df_bt.apply(pd.to_numeric, errors='coerce').dropna()
        
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000)
        cerebro.broker.setcommission(commission=0.001)
        
        # Add data feed
        feed = PandasRegimeFeed(
            dataname=df_bt,
            datetime=None,  # Use index as datetime
            open=0, high=1, low=2, close=3, volume=4, openinterest=5,
            regime_code=6
        )
        cerebro.adddata(feed)
        
        # Add strategy and analyzers
        cerebro.addstrategy(RegimeStrategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run simulation
        results = cerebro.run()
        strat = results[0]
        
        return {
            "initial_value": 100000,
            "final_value": round(cerebro.broker.getvalue(), 2),
            "total_return": round(((cerebro.broker.getvalue() - 100000) / 100000) * 100, 2),
            "sharpe_ratio": round(strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0, 3),
            "max_drawdown": round(strat.analyzers.dd.get_analysis().get('max', {}).get('drawdown', 0), 3),
            "total_trades": strat.analyzers.trades.get_analysis().get('total', {}).get('closed', 0),
            "winning_trades": strat.analyzers.trades.get_analysis().get('won', {}).get('total', 0),
            "losing_trades": strat.analyzers.trades.get_analysis().get('lost', {}).get('total', 0)
        }
    except Exception as e:
        logger.error(f"Error in Backtrader simulation: {str(e)}")
        raise ValueError(f"Backtrader simulation failed: {str(e)}")

def calculate_simple_strategy_returns(data):
    try:
        # Work with a copy
        data = data.copy()
        
        # Verify and create required columns if missing
        if 'log_return' not in data.columns:
            data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        if 'regime_label' not in data.columns:
            raise ValueError("Missing regime labels - cannot calculate strategy returns")
        
        # Clean data
        data = data.dropna(subset=['log_return', 'regime_label']).copy()
        
        def strategy_logic(row):
            label = str(row['regime_label'])
            if label == 'Bull':
                return row['log_return']
            elif label == 'Bear':
                return -row['log_return']
            elif label == 'Volatile':
                return 0.5 * row['log_return']
            else:  # Sideways or unknown
                return 0

        data['strategy_return'] = data.apply(strategy_logic, axis=1)
        data['strategy_cum'] = (1 + data['strategy_return']).cumprod()
        data['buy_hold_cum'] = (1 + data['log_return']).cumprod()

        return data

    except Exception as e:
        logger.error(f"Error calculating strategy returns: {str(e)}")
        raise

def create_visualizations(data, model, scaler, features):
    try:
        # Create a copy to avoid modifying original data
        plot_data = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Close']
        for col in required_cols:
            if col not in plot_data.columns:
                raise ValueError(f"Required column '{col}' not found in data. Available columns: {plot_data.columns.tolist()}")
        
        # Create price chart with regime labels
        plot_data_reset = plot_data.reset_index()
        
        # Get the first column name (should be the date index)
        date_col = plot_data_reset.columns[0]
        
        fig_price = px.line(
            plot_data_reset,
            x=date_col,
            y='Close',
            color='regime_label' if 'regime_label' in plot_data.columns else None,
            title='Regime-Labeled Price Data'
        )
        
        # Prepare features for scaling - ensure all features exist
        available_features = [f for f in features if f in plot_data.columns]
        if not available_features:
            raise ValueError(f"No features found in data. Required: {features}, Available: {plot_data.columns.tolist()}")
        
        features_data = plot_data[available_features].copy()
        
        # Scale features
        X_scaled = pd.DataFrame(
            scaler.transform(features_data), 
            index=plot_data.index, 
            columns=available_features
        )
        
        posteriors = model.predict_proba(X_scaled)
        posterior_df = pd.DataFrame(
            posteriors, 
            index=plot_data.index, 
            columns=[f'Regime {i}' for i in range(model.n_components)]
        )

        
        # Create posterior probabilities chart
        posterior_reset = posterior_df.reset_index()
        regime_cols = [col for col in posterior_reset.columns if col.startswith('Regime')]
        
        fig_post = px.area(
            posterior_reset, 
            x=posterior_reset.columns[0],  # Use index column
            y=regime_cols,
            title="Posterior Probabilities"
        )
        
        # Performance comparison - check if performance columns exist
        performance_columns = ['strategy_cum', 'buy_hold_cum']
        available_perf_cols = [col for col in performance_columns if col in plot_data.columns]
        
        if available_perf_cols:
            performance_df = plot_data[available_perf_cols].copy()
            
            # Rename columns for better display
            column_mapping = {
                'strategy_cum': 'Regime Strategy',
                'buy_hold_cum': 'Buy & Hold'
            }
            performance_df = performance_df.rename(columns=column_mapping)
            
            performance_reset = performance_df.reset_index()
            perf_cols = [col for col in performance_reset.columns if col in ['Regime Strategy', 'Buy & Hold']]
            
            fig_perf = px.line(
                performance_reset,
                x=performance_reset.columns[0],
                y=perf_cols,
                title="Strategy vs Buy & Hold Performance"
            )
        else:
            # Create empty chart if performance data not available
            fig_perf = px.line(title="Strategy vs Buy & Hold (Performance data not available)")
            fig_perf.add_annotation(
                text="Performance data not calculated yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return {
            'price_chart': pio.to_json(fig_price),
            'posterior_chart': pio.to_json(fig_post),
            'performance_chart': pio.to_json(fig_perf)
        }
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.error(f"Data shape: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
        logger.error(f"Data columns: {data.columns.tolist() if hasattr(data, 'columns') else 'Unknown'}")
        logger.error(f"Column types: {type(data.columns) if hasattr(data, 'columns') else 'Unknown'}")
        logger.error(f"Features: {features}")
        raise