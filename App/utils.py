from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np

hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0:"SOS", 1:"EOS"}
    self.n_words = 2
  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)
  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )

def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
  return s.strip()

MAX_LEN = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
  indexes = indexesFromSentence(lang, sentence)
  indexes.append(EOS_token)
  return torch.tensor(indexes, dtype=torch.long, device=device).view(1,-1)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
  with torch.no_grad():
    input_tensor = tensorFromSentence(input_lang, sentence)

    encoder_outputs, encoder_hidden = encoder(input_tensor)

    decoder_outputs, decoder_hidden, decoder_attention = decoder(encoder_outputs, encoder_hidden)

    _, topi = decoder_outputs.topk(1)

    decoded_ids = topi.squeeze()
    decoded_words = []

    for idx in decoded_ids:
      if idx.item()==EOS_token:
        decoded_words.append('<EOS>')
        break
      else:
        decoded_words.append(output_lang.index2word[idx.item()])
  return decoded_words, decoder_attention

class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size, dropout_p=0.1):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, input):
    embedded = self.dropout(self.embedding(input))
    # print('embedded :', embedded.shape)
    output, hidden = self.gru(embedded)
    # print('output :', output.shape)
    # print('Hidden :', hidden[0].shape)
    return output, hidden
  
class BahdanauAttention(nn.Module):
  def __init__(self, hidden_size):
    super(BahdanauAttention,self).__init__()
    self.Wa = nn.Linear(hidden_size, hidden_size)
    self.Ua = nn.Linear(hidden_size, hidden_size)
    self.Va = nn.Linear(hidden_size, 1)

  def forward(self, query, keys):
    # print(self.Wa(query).shape)
    # print(self.Ua(keys).shape)
    scores = self.Va(torch.tanh( self.Wa(query) + self.Ua(keys) ))
    # print('Scores :', scores.shape)
    scores = scores.squeeze(2).unsqueeze(1)
    # print('Scores :', scores.shape)

    weights = F.softmax(scores, dim = -1)
    # print('Weights : ', weights.shape)
    context = torch.bmm(weights, keys)
    # print('context : ', context.shape)
    return context, weights
  
class AttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.embedding = nn.Embedding(output_size, hidden_size)
    self.attention = BahdanauAttention(hidden_size)
    self.gru = nn.GRU(2*hidden_size, hidden_size, batch_first=True)
    self.out = nn.Linear(hidden_size, output_size)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
    batch_size = encoder_outputs.size(0)
    decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
    # print('initial_decoder_inp : ', decoder_input.shape)
    decoder_hidden = encoder_hidden
    # print('decoder_hidden:', decoder_hidden.shape)
    decoder_outputs = []
    attentions = []
    for i in range(MAX_LEN):
      decoder_output, decoder_hidden, attn_weights = self.forward_step(
          decoder_input, decoder_hidden, encoder_outputs
      )
      decoder_outputs.append(decoder_output)
      attentions.append(attn_weights)

      if target_tensor is not None:
        decoder_input = target_tensor[:,i].unsqueeze(1)

      else:
        _, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze(-1).detach()

    decoder_outputs = torch.cat(decoder_outputs, dim=1)
    decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
    attentions = torch.cat(attentions, dim=1)

    return decoder_outputs, decoder_hidden, attentions
  def forward_step(self, inp, hidden, encoder_outputs):
    # print('inp : ', inp.shape)
    embedded = self.dropout(self.embedding(inp))
    # print('hidden : ', hidden.shape)
    query = hidden.permute(1,0,2)
    # print('query : ', query.shape)
    # print('enc_out : ', encoder_outputs.shape)
    context, attn_weights = self.attention(query, encoder_outputs)
    input_gru = torch.cat((embedded, context), dim=2)
    # print('input_gru :', input_gru.shape)
    output, hidden  = self.gru(input_gru, hidden)
    output = self.out(output)

    return output, hidden, attn_weights
