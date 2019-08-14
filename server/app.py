import os
from flask import Flask, jsonify, request, json, render_template
from predictor import Predictor

app = Flask(__name__, static_folder='static', template_folder='template')


@app.route('/predict', methods=['POST'])
def predict_word():
    data = request.json
    if 'input' in data:
        print('Predicting...')
        inp = data['input']
        fl = inp.split()
        if len(fl) == 2: # predict last name
            last_name = fl[1]
            res = predictor.predict_last_name(last_name)
            res = {inp+x[0]:x[1] for x in res}
        else:
            res = {}
        res = json.dumps(res, ensure_ascii=False)
        return res
    else:
        return "error"

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    predictor = Predictor()
    fn_model_path = 'weight/fn_model.h5'
    ln_model_path = 'weight/ln_model.h5'
    fn_vocab_path = 'weight/fn_tokenizer.h5'
    ln_vocab_path = 'weight/ln_tokenizer.h5'
    predictor.init_model(fn_model_path, ln_model_path, fn_vocab_path, ln_vocab_path)
    port = int(os.environ.get('PORT', 5000))
    host = 'localhost'
    app.run(host=host, port=port)
