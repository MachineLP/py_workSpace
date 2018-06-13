# coding=utf-8
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import collections
import json

# 以下测试用
if __name__ == '__main__':
    File = open("sample_res.csv", "r")

    id_list = []
    for line in File:
        try:
            img_url= line.split(' ')
            sample_id = img_url[0]
            # if sample_id not in id_list:
            id_list.append(sample_id)

        except:
            continue
    File.close()

    counter = collections.Counter(id_list)  
    print (counter)
    print (len(counter.items()))

    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    #  [('580', 1), ('428', 1)]
    print (count_pairs)

    sample_id_list = []
    sample_id_num = []
    for pair in count_pairs:
        sample_id_list.append(int(pair[0]))
        print (pair[0])
        sample_id_num.append(int(pair[1]))
        print (pair[1])

    X=sample_id_list
    Y=sample_id_num
    fig = plt.figure()
    plt.bar(X,Y,0.8,color="green")
    plt.xlabel("sample_id")
    plt.ylabel("sample_id_num")
    plt.title("bar")

    plt.savefig("barChart.jpg")

    plt.show()




 
