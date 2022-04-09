from crypt import methods
from urllib import response
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

import inference

app = Flask(__name__,
            static_folder="../dist/static",
            template_folder="../dist")
CORS(app)

bert = inference.Inference()


@app.route('/api/ner', methods=["GET", "POST"])
def random_number():
    text = '日本の首都は東京都で、総理大臣は岸田総理です。'
    if request.method == 'POST':
        text = request.json["input_text"]
        print(text)
    response = bert.inference(text)
    return jsonify({'result': response})


@app.route('/')
def hello_world():
    # return str(ans)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
