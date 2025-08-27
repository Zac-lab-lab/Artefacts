# Spam Classifier Web Application

A web-based spam detection system that allows users to input email messages and receive real-time classification feedback.

## Features

- **Real-time Classification**: Instantly classify emails as spam or legitimate (ham)
- **Confidence Scoring**: Shows confidence percentage for each prediction
- **Modern UI**: Clean, responsive web interface
- **Example Messages**: Pre-loaded examples to test the system
- **Mobile Friendly**: Works on desktop and mobile devices

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**:
   Open your browser and go to: `http://localhost:5000`

## How to Use

1. **Enter Email Text**: Paste or type an email message in the text area
2. **Click Classify**: Press the "Classify Email" button
3. **View Results**: See the classification result and confidence score
4. **Try Examples**: Use the example buttons to test different types of messages

## Classification Results

- **Spam**: Red background with confidence percentage
- **Not Spam (Ham)**: Green background with confidence percentage
- **Loading**: Blue background while processing

## Technical Details

- **Backend**: Flask web framework
- **ML Model**: Naive Bayes classifier with TF-IDF vectorization
- **Features**: Text preprocessing with business indicators and spam signals
- **Model Training**: Uses an expanded dataset with edge cases and previously misclassified examples

## File Structure

```
spam classifer/
├── app.py              # Flask web application
├── spam.py             # Spam classifier model and preprocessing
├── templates/
│   └── index.html      # Web interface template
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Model Performance

The classifier has been trained on an expanded dataset that includes:
- Common spam patterns (prizes, money offers, urgent requests)
- Legitimate business communications
- Edge cases and previously misclassified examples
- Ambiguous messages for better generalization

## API Endpoint

The application provides a REST API endpoint:

- **POST** `/classify`
- **Body**: `{"email_text": "your email message here"}`
- **Response**: 
  ```json
  {
    "success": true,
    "classification": "Spam",
    "confidence": "85.2% confident this is spam",
    "confidence_percentage": 85.2,
    "is_spam": true
  }
  ```

## Troubleshooting

- **Model not found**: The application will automatically train the model if `spam_detection_model.joblib` doesn't exist
- **Port already in use**: Change the port in `app.py` or kill the process using port 5000
- **Dependencies missing**: Run `pip install -r requirements.txt`

## Security Notes

- This is a demonstration application
- Do not use for production without proper security measures
- The model is trained on a limited dataset and may not catch all spam types
- Always validate and sanitize user inputs in production environments 