#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:24:55 2017

@author: liupeng
"""

import collections  
import numpy as np  
import tensorflow as tf  
   
#-------------------------------数据预处理---------------------------#  
   
poetry_file ='poetry.txt'  
   
# 诗集  
poetrys = []  
# with open(poetry_file, "r", encoding='utf-8',) as f:
with open(poetry_file, "r") as f: 
    for line in f:  
        try:  
            title, content = line.strip().split(':')  
            content = content.replace(' ','')  
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:  
                continue  
            if len(content) < 5 or len(content) > 79:  
                continue  
            content = '[' + content + ']'  
            poetrys.append(content)  
        except Exception as e:   
            pass  
   
# 按诗的字数排序  
poetrys = sorted(poetrys,key=lambda line: len(line))  
print('唐诗总数: ', len(poetrys))  
   
# 统计每个字出现次数  
all_words = []  
for poetry in poetrys:  
    all_words += [word for word in poetry]  
counter = collections.Counter(all_words)  
count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
words, _ = zip(*count_pairs)  
   
# 取前多少个常用字  
words = words[:len(words)] + (' ',)  
# 每个字映射为一个数字ID  
word_num_map = dict(zip(words, range(len(words))))  
# 把诗转换为向量形式，参考TensorFlow练习1 
to_num = lambda word: word_num_map.get(word, len(word))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]


# 训练取64首诗进行训练
batch_size = 64
n_chunk = len(poetrys_vector)  
x_batches = []
y_batches = []

for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    
    batches = poetrys_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:,:-1] = xdata[:,1:]
    x_batches.append(xdata)
    y_batches.append(ydata)
































