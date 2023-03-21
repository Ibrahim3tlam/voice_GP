from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

model= joblib.load('disable_model.pkl')
vectorizer =joblib.load('disable_preproccess.pkl')

@app.route("/voiceApi", methods=["POST"])
def num():
    try:
        user_text = request.args.get('message')
        vector = vectorizer.transform([user_text])
        prediction = model.predict(vector)
        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        })


if __name__ == "__main__":
    app.run(debug=True, port=200)