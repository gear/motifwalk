# coding: utf-8
text8_url = 'http://mattmahoney.net/dc/text8.zip'
import os
import numpy as np
import tensorflow as tf
from six.moves import urllib
from six.moves import urllib, xrange
import zipfile
import random
import collections
from __future__ import division
from __future__ import print_function
get_ipython().magic(u'ls ')
filepath = os.path('./data/')
filepath = ('./data/text8.zip')
os.path.exists(filepath)
os.path.exists(filepath[:-9])
zipfile = urllib.request.urlretrieve(text8_url, filepath) 
zipfile
filepath
statinfo = os.stat(filepath)
statinfo
with zipfile.ZipFile(filepath) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
del zipfile
import zipfile
with zipfile.ZipFile(filepath) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
data
len(data)
vocabulary_size = 50000
count = [['UNK', -1]]
count
count.extend(collections.Counter(data).most_common(vocabulary_size - 1))
len(count)
count[0]
count[1]
type(count)
dictionary = dict()
count[2]
count[3]
type(count)
len(dictionary)
for word, _ in count:
    dictionary[word] = len(dictionary)
dictionary('the')
dictionary['the']
dictionary['UNK']
dictionary['of']
zip(1,2)
zip([1,2], 3)
zip([1,2], [3, 4])
zip([1,2,5], [3, 4])
zip([1,2,5], [3, 4])
raw_data = list()
raw_data.extend(data)
len(raw_data)
len(data)
data = list()
len(data)
unk_count = 0
for word in raw_data:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0
        unk_count += 1
    data.append(index)
data[1]
data[0]
count[0]
count[0][1] = unk_count
count[0][1]
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
reverse_dictionary
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
os.sys('git status')
os.sys('ls')
os.system('ls')
os.system('ls')
os.system('pwd')
get_ipython().magic(u'ls ')
os.getcwd()
get_ipython().magic(u"save '~/Dropbox/la.ipython_log'")
get_ipython().magic(u'save ~/Dropbox/la.ipython_log')
get_ipython().magic(u'save')
get_ipython().magic(u'save la.ipython_log')
get_ipython().magic(u'save la 1-78')
get_ipython().magic(u"save 'la' 1-78")
get_ipython().magic(u"save '~/Dropbox/la' 1-78")
get_ipython().magic(u"save '~la' 1-78")
get_ipython().magic(u"save '~/la' 1-78")
os.chdir('~/Dropbox/WorkingFiles/motifwalk/research/src/2016-06-27-word2vec_tensorflow/')
import readline
readline.write_history_file('/home/hoangnt/Dropbox/WorkingFiles/motifwalk/research/src/2016-06-27-word2vec_tensorflow/ipython_log')
os.chdir('/home/hoangnt/Dropbox')
os.chdir('./WorkingFiles/motifwalk/research/src/2016-06-27-word2vec_tensorflow')
get_ipython().magic(u'ls ')
get_ipython().magic(u'save la 1-90')
