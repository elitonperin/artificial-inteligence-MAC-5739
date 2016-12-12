#all imports
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
#from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import subprocess
import time
from datetime import datetime, timedelta
import pickle
import string


#Tratamento para retirar pontuacao e maiusculas
def small(x):
    return chr(ord(x)+32)

alphabeto ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

dic_trans = {key: None for key in string.punctuation}
for letter in alphabeto:
    dic_trans[letter] = small(letter)

translator = str.maketrans(dic_trans)

#carregando o texto
words = []
with open('Wiki_pt.txt') as inputfile:
    for line in inputfile:
        aux = line.translate(translator)
        words.extend(aux.strip().split())

print('numero total de palavras {0}'.format(len(words)))

#escolhemos um tamanho V para o vocabulario
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]

  #pegamos as V-1 palavras do texto mais utilizadas
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

  #atribuimos para cada uma das V-1 palavras mais usadas
  #um numero unico
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)

  #reconstruimos o texto como uma lista de indices
  #para cada palavra w, se w in dictionary, seu indice eh o
  #mesmo que o do dicionario, caso contratio seu indice
  #eh 0 para 'UNK'.
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count

  #reverse_dict é o dicionario com o mesmo conteudo que 
  #dictionary mas invertido, i.e, se palavra:index in dictionary,
  #entao index:palavra in reverse_dictionary
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  

data_index = 0

#funcao que gera as instancias de treinamento dados
#o tamanho do batch, a quantidade de vezes que olhamos
#para uma palavra e o tamano do contexto
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels



batch_size = 128
embedding_size = 128 
skip_window = 1 
num_skips = 2 

valid_size = 16 
valid_window = 100 
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 

graph = tf.Graph()

with graph.as_default():

  train_dataset = tf.placeholder(tf.int32, shape=[batch_size],name= 'X')
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1],name= 'Y')
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name ='valid_dataset')
  
  # Variaveis.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embedding_matrix')
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)), name='softmax_weights')
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]), name ='softmax_biases')

  tf.histogram_summary('Embeddings',embeddings)
  tf.histogram_summary('Softmax_weights',softmax_weights)
  
  # Modelo

  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
 
  #funcao de erro
  with tf.name_scope('loss'): 
    loss = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                 train_labels, num_sampled, vocabulary_size))
    tf.scalar_summary(loss.op.name,loss) #write loss to log


  with tf.name_scope('training'):
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


log_basedir = 'logs'
run_label = time.strftime('%d-%m-%Y_%H-%M-%S')
log_path = os.path.join(log_basedir,run_label)


num_steps = 100001
initial_time = time.time()

with tf.Session(graph=graph) as session:
  summary_writer = tf.train.SummaryWriter(log_path, session.graph) 
  all_summaries = tf.merge_all_summaries() 
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):


    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}

    start_time = time.time()
    _, l,summary = session.run([optimizer, loss,all_summaries], feed_dict=feed_dict)
    duration = time.time() - start_time

    summary_writer.add_summary(summary,step)
    summary_writer.flush()

    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      print('Average loss at step %d: %f' % (step, average_loss))
      print('Duration: %.3f sec' % duration)
      average_loss = 0
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 3 
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
      print('Duration: %.3f sec' % duration)
  final_embeddings = normalized_embeddings.eval()


general_duration = time.time() - initial_time
sec = timedelta(seconds=int(general_duration))
d_time = datetime(1,1,1) + sec
print(' ')
print('The duration of the whole training with % s steps is %.2f seconds,'\
      % (num_steps,general_duration))
print("which is equal to:  %d:%d:%d:%d" % (d_time.day-1, d_time.hour, d_time.minute, d_time.second), end='')
print(" (DAYS:HOURS:MIN:SEC)")
print(' ')
print(log_path)


f = open(run_label + '_teste.pickle', 'wb')
di ={'dictionary':dictionary, 'data':data, 'count':count, 'reverse_dictionary':reverse_dictionary, 'final_embeddings':final_embeddings}
pickle.dump(di,f)
f.close()
