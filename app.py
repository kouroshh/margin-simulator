from flask import Flask, request, jsonify
import functools
import pandas as pd
from src_margins.core import calculate
from flasgger import Swagger
from utilities import read_arrow

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
    output = calculate(df)
    # Show the DataFrame
    print(output)
    # Check if the data was provided
    if not data:
        return jsonify({"message": "No JSON payload provided"}), 400

    # Here, you can process the data (for now, just return it)
    #return jsonify({"received_data": output})
    return output

def convert_to_dataframe(json_data):
    # Create empty lists to store the data
    portfolio_nb = []
    isin = []
    prod_curcy = []
    qty = []
    trade_price = []
    
    # Iterate over portfolios in the JSON
    for portfolio_id, portfolio_data in json_data.items():
        # Extract the portfolio number (assuming it's like 'portfolio_nb_1' -> '1')
        portfolio_num = portfolio_id.split('_')[2]
        
        # Iterate over ISINs in each portfolio
        for isin_id, isin_data in portfolio_data.items():
            # Append the data to the respective lists
            portfolio_nb.append(portfolio_num)
            isin.append(isin_id)
            prod_curcy.append(isin_data['prod_curcy'])
            qty.append(isin_data['qty'])
            trade_price.append(isin_data['trade_price'])
    
    # Create the DataFrame from the lists
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
