from flask import Flask, jsonify, request
import pandas as pd
import random
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

app = Flask(__name__)
base_input_path = r"input"
base_output_path = r"output"


def get_consecutive_data_points(exchange: str, stock_id: str) -> dict:
    """
        Function to get 10 consecutive entries for given stock.
        :param exchange: Name of the exchange (NASDAQ, LSE, etc.)
        :param stock_id: Stock id (TSLA, GSK, etc.)
        :return: List of entries
    """
    file_path = os.path.join(base_input_path, exchange, f"{stock_id}.csv")
    try:
        df = pd.read_csv(file_path)
        start_index = random.randint(0, len(df) - 10)
        consecutive_data = df.iloc[start_index: start_index + 10]
        return {'status': 'success', 'data': consecutive_data}
    except FileNotFoundError as e:
        return {'status': 'fail', 'error': f'get_consecutive_data_points: {e}'}


def predict_next_3_values(consecutive_data):
    """
        Function to predict next 3 prices in a timeseries, given the data points
        :param consecutive_data: The stock data that we train prediction on - Type Dataframe
        :return: List of predicted prices, List of timeseries for the predicted prices
    """

    stock_prices = consecutive_data.iloc[:, 2].values
    stock_dates = consecutive_data.iloc[:, 1].values

    # convert dates to number
    stock_dates = pd.to_datetime(stock_dates, dayfirst=True)

    # normalize dates
    stock_dates_numeric = (stock_dates - stock_dates.min()) / np.timedelta64(1, 'D')
    stock_dates_numeric = stock_dates_numeric.values.reshape(-1, 1)  # Reshaping for sklearn

    model = LinearRegression()
    model.fit(stock_dates_numeric, stock_prices)

    future_dates = [stock_dates[-1] + timedelta(days=i) for i in range(1, 4)]
    future_dates_numeric = [(date - stock_dates[0]) / np.timedelta64(1, 'D') for date in future_dates]
    future_dates_numeric = np.array(future_dates_numeric).reshape(-1, 1)
    future_dates_numeric = np.array(future_dates_numeric).reshape(-1, 1)

    future_prices = model.predict(future_dates_numeric)
    return future_prices, future_dates


def write_output_to_csv(exchange: str, stock_id: str, output_data):
    """
       Simple helper to write to a csv file
    """
    try:
        output_file_path = os.path.join(base_output_path, exchange, f"{stock_id}_output.csv")
        output_df = pd.DataFrame(output_data, columns=['Stock-ID', 'Timestamp', 'stock_price'])
        output_df.to_csv(output_file_path, index=False)
        return {"status": "success"}

    except FileNotFoundError as e:
        return {"status": "fail", "error": f'write_output_to_csv: {e}'}


@app.route('/api/get_consecutive_data/<exchange>/<stock_id>', methods=['GET'])
def get_consecutive_data(exchange: str, stock_id: str):
    result = get_consecutive_data_points(exchange, stock_id)
    if result['status'] == 'fail':
        return result, 400
    consecutive_data = result['data']
    return consecutive_data.to_dict(orient='records')


@app.route('/api/predict_next_3_values', methods=['POST'])
def predict_next_3_values_api():
    data = request.json
    exchange = data.get('exchange')
    stock_id = data.get('Stock-ID')

    result = get_consecutive_data_points(exchange, stock_id)
    if result['status'] == 'fail':
        return result, 400

    consecutive_data = result['data']
    predicted_values, future_dates = predict_next_3_values(consecutive_data)

    # Prepare the output data
    output_data = []
    for i, value in enumerate(predicted_values):
        output_data.append({
            'Stock-ID': stock_id,
            'Timestamp': future_dates[i].strftime("%d-%m-%Y"),
            'stock_price': value
        })

    # Write the output to a CSV file
    result = write_output_to_csv(exchange, stock_id, output_data)
    if result['status'] == 'fail':
        return result, 400

    return {"status": "success", "predicted_values": output_data}


if __name__ == '__main__':

    # Go and test with all possible input
    for dirpath, _, filenames in os.walk(base_input_path):

        if dirpath == base_input_path:
            continue

        exchange = os.path.basename(dirpath)
        os.makedirs(os.path.join(base_output_path, exchange), exist_ok=True)
        for filename in filenames:
            if filename.startswith('.'):
                continue
            stock_id = filename.split('.')[0]
            get_response = app.test_client().get(f'/api/get_consecutive_data/{exchange}/{stock_id}')
            print(f'GET request for {exchange}/{stock_id} - Status Code: {get_response.status_code}')

            # Send POST request for each stock ID
            post_data = {"exchange": exchange, "Stock-ID": stock_id}
            post_response = app.test_client().post('/api/predict_next_3_values', json=post_data)
            print(f'POST request for {exchange}/{stock_id} - Status Code: {post_response.status_code}')

    # Test error handling
    get_response = app.test_client().get(f'/api/get_consecutive_data/random/thing')
    print(get_response.status_code)
    print(get_response.json)

    app.run()
