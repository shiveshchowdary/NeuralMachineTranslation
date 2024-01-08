from flask import Flask, render_template, request
from utils import EncoderRNN, AttnDecoderRNN, evaluate, Lang, hidden_size, device
import torch

import pickle
with open('model/input_lang.pkl','rb') as f:
    input_lang = pickle.load(f)

with open('model/output_lang.pkl','rb') as f:
    output_lang = pickle.load(f)

# print(input_lang.n_words)
# print(output_lang.n_words)


encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

encoder.load_state_dict(torch.load('model/encoder.pth',map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('model/decoder.pth',map_location=torch.device('cpu')))

# test_sentence = 'il est americain'
# output_words, _ = evaluate(encoder, decoder, test_sentence, input_lang, output_lang)
# output_sentence = ' '.join(output_words[:-1])
# print(output_sentence)

# Set the models to evaluation mode
encoder.eval()
decoder.eval()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ['GET','POST'])
def predict():
    test_sentence = request.form.get('input_text')#'il est americain'
    output_words, _ = evaluate(encoder, decoder, test_sentence.lower(), input_lang, output_lang)
    output_sentence = ' '.join(output_words[:-1])
    # print(output_sentence)
    return render_template("predict.html", prediction_text = test_sentence + ' ----> '+ output_sentence)


if __name__ == '__main__':
    app.run(debug=True)