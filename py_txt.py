# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""
import sys

fr = open('xyj.txt', 'r', encoding='UTF-8')

characters = []
stat = {}

for line in fr:
    # 去掉每一行两边的空白
    line = line.strip()

    # 如果为空行则跳过该轮循环
    if len(line) == 0:
        continue

    # 将文本转为unicode，便于处理汉字
    line = str(line)
    # print (line)

    # 遍历该行的每一个字
    for x in range(len(line)):
        # 去掉标点符号和空白符
        if line[x] in [' ', '\t', '\n', '。', '，', '(', ')', '（', '）', '：', '□', '？', '！', '《', '》', '、', '；', '“', '”', '……']:
            continue

        # 尚未记录在characters中
        if not line[x] in characters:
            characters.append(line[x])

        # 尚未记录在stat中
        if not line[x] in stat:
            stat[line[x]] = 0
        # 汉字出现次数加1
        stat[line[x]] += 1

print (len(characters))
print (len(stat))

def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst
# lambda生成一个临时函数
# d表示字典的每一对键值对，d[0]为key，d[1]为value
# reverse为True表示降序排序
stat = sorted(dict2list(stat), key=lambda d:d[1], reverse=True)

fw = open('result.csv', 'w', encoding='UTF-8')
for item in stat:
    # 进行字符串拼接之前，需要将int转为str
    # 字典的遍历方式: fw.write(item + ',' + str(state[item]) + '\n')
    # 排完序后是列表
    fw.write(item[0] + ',' + str(item[1]) + '\n')

fr.close()
fw.close()

print ("success!")
