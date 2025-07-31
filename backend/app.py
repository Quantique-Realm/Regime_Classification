from flask import Flask, request, jsonify
from flask_cors import CORS
from model_core import *  # imports all model functions

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        use_backtrader = data.get('use_backtrader', False)

        # 1. Get and process data
        df = get_market_data(ticker, start_date, end_date)
        df = engineer_features(df)

        # 2. Train HMM
        features = ['log_return', 'rsi', 'macd', 'macd_diff', 'rolling_mean', 'bb_width', 'atr']
        model, scaler, regimes, idx = train_hmm_model(df, features)

        # 3. Label regimes
        df_labeled, regime_labels = label_regimes(df, regimes, idx)

        # 4. Backtrader simulation
        bt_stats = run_backtrader_simulation(df_labeled) if use_backtrader else {}

        # 5. Regime-based strategy returns
        df_labeled = calculate_simple_strategy_returns(df_labeled)

        # 6. Create charts
        charts = create_visualizations(df_labeled, model, scaler, features)

        return jsonify({
            'fig_price': charts['price_chart'],
            'fig_post': charts['posterior_chart'],
            'fig_perf': charts['performance_chart'],
            'stats': bt_stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
