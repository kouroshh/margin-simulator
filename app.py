from flask import Flask, request, jsonify
import functools
import pandas as pd
from src_margins.core import calculate
from flasgger import Swagger
from utilities import read_arrow
import os
import json

app = Flask(__name__)
swagger = Swagger(app)

# Secret key to verify API key/token
SECRET_API_KEY = "margin-simulator-key"

# Simple function to check if the provided API key is correct
def api_key_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from headers
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != SECRET_API_KEY:
            return jsonify({'message': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/token', methods=['GET'])
def get_token():
    # For simplicity, just return the API key as a "token"
    return jsonify({"token": SECRET_API_KEY})

@app.route('/protected', methods=['GET'])
@api_key_required  # Protect this route with API key
def protected():
    return jsonify({"message": "Hello, you are authorized!"})

@app.route('/calculate_margin', methods=['POST'])
@api_key_required  # Protect this route with API key
def process_data():
    """
    A Method to calculate the margin
    ---
    responses:
      200:
        description: A successful response
        content:
          application/json:
            example: {"message": "Hello, World!"}
    """
    # Get the JSON payload from the incoming request
    data = request.get_json()
    df = convert_to_dataframe(data)
    output = calculate(df, data["dateTo"])
    # Show the DataFrame
    
    # Check if the data was provided
    if not data:
        return jsonify({"message": "No JSON payload provided"}), 400

    # Here, you can process the data (for now, just return it)
    output = json.loads(output)
    portfolios = []
    for index in output["portfolio_nb"]:
        portfolio = {
            "portfolio_nb": output["portfolio_nb"][index],
            "ES": output["ES"][index],
            "DECO": output["DECO"][index],
            "whatif": output["whatif"][index],
            "mtm": output["mtm"][index],
            "initial_margin": output["initial_margin"][index],
            "gross_pos_value": output["gross_pos_value"][index],
            "margin_%": output["margin_%"][index]
        }
        portfolios.append(portfolio)
    #return jsonify({"received_data": output})
    return portfolios

def convert_to_dataframe(json_data):
    # Create empty lists to store data
    portfolio_nb = []
    isin = []
    prod_curcy = []
    qty = []
    trade_price = []

    # Loop through portfolios in the JSON data
    for portfolio in json_data['portfolios']:
        portfolio_title = portfolio['title']  # Portfolio title
        
        # Loop through positions in each portfolio
        for position in portfolio['positions']:
            portfolio_nb.append(portfolio_title)
            isin.append(position['isin'])
            prod_curcy.append(position['currency'])
            qty.append(int(position['quantity']))  # Convert quantity to int
            trade_price.append(position['tradingPrice'])  # Trading price is directly added

    # Convert the lists into a DataFrame
    df = pd.DataFrame({
        'portfolio_nb': portfolio_nb,
        'isin': isin,
        'prod_curcy': prod_curcy,
        'qty': qty,
        'trade_price': trade_price
    })
    
    return df

if __name__ == '__main__':
    app.run(debug=True)
