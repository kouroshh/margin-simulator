from flask import Flask, request, jsonify
import functools

app = Flask(__name__)

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
    # Get the JSON payload from the incoming request
    data = request.get_json()

    # Check if the data was provided
    if not data:
        return jsonify({"message": "No JSON payload provided"}), 400

    # Here, you can process the data (for now, just return it)
    return jsonify({"received_data": data})

if __name__ == '__main__':
    app.run(debug=True)
