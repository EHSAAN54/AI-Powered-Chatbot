from flask import Flask, request, jsonify
from chatbot import Chatbot

app = Flask(__name__)
chatbot = Chatbot()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    response = chatbot.predict(user_message)
    sentiment = chatbot.get_sentiment(user_message)

    return jsonify({
        'response': response,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
