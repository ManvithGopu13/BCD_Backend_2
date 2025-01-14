from flask import Flask, request, jsonify
import pickle
import numpy as np  

# Load the model
with open('breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the request
        data = request.json
        features = np.array([[data['radius'], data['texture'], data['perimeter'], data['area'], data['smoothness']]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        condition = 'Malignant' if prediction == 1 else 'Benign'
        
        # Log details of the request and prediction
        print(f"Received data: {data}")
        print(f"Prediction: {condition}")
        
        return jsonify({'condition': condition})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

 # if __name__ == '__main__':
#     # Run the Flask app, allowing access via your device's IP address
#     app.run(host='172.20.10.3', debug=True, port=5001, use_reloader=False)

