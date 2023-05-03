from flask import Flask, jsonify, request
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity(user_input):
    X_user = vectorizer.transform([user_input])

    # Set a similarity threshold
    similarity_threshold = 0.3

    # Compute the cosine similarity between user input and text data
    similarity_scores = cosine_similarity(X_user, X)

    # Check if the highest similarity score is above the threshold
    if similarity_scores.max() > similarity_threshold:
        # If so, predict the corresponding target value
        y_user = model.predict(X_user)
        result = y_user[0]
    else:
        # Otherwise, classify as "no"
        result = 0

    return result


app = Flask(__name__)

model= joblib.load('disable_model.pkl')
vectorizer =joblib.load('disable_preproccess.pkl')
X = joblib.load('data.pkl')


@app.route("/voiceApi", methods=["POST"])
def num():
    try:
        user_text = request.args.get('message')

        prediction = get_similarity(user_text)

        return jsonify({
            'prediction': int(prediction)
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        })


if __name__ == "__main__":
    app.run(debug=True, port=200)