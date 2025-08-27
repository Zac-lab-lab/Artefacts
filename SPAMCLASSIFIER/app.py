from flask import Flask, render_template, request, jsonify
from spam import preprocess_text
from joblib import load
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "spam_detection_model.joblib")
if os.path.exists(model_path):
    pipeline = load(model_path)
else:
    pipeline = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '').strip()
        
        if not email_text:
            return jsonify({
                'success': False,
                'error': 'Please provide email text'
            })
        
        if pipeline is None:
            return jsonify({
                'success': False,
                'error': 'Model not found. Please train the model by running spam.py.'
            })
        
        # Preprocess the email text
        processed_text = preprocess_text(email_text)
        
        # Make prediction
        prediction = pipeline.predict([processed_text])[0]
        
        # Get prediction probability
        proba = pipeline.predict_proba([processed_text])[0]
        confidence = max(proba) * 100
        
        # Determine classification
        if prediction == 1:
            classification = "Spam"
            confidence_text = f"{confidence:.1f}% confident this is spam"
        else:
            classification = "Not Spam (Ham)"
            confidence_text = f"{confidence:.1f}% confident this is legitimate"
        
        return jsonify({
            'success': True,
            'classification': classification,
            'confidence': confidence_text,
            'confidence_percentage': round(confidence, 1),
            'is_spam': bool(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing request: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5025) 