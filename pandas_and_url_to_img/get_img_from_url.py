# coding=utf-8
import pandas as pd  
import json
import os
from urllib import request

def save_img(img_url,file_name,file_path='img'):
    #保存图片到磁盘文件夹 file_path中，默认为当前脚本运行目录下的 book\img文件夹
    try:
        if not os.path.exists(file_path):
            print ('文件夹',file_path,'不存在，重新建立')
            #os.mkdir(file_path)
            os.makedirs(file_path)
        #获得图片后缀
        file_suffix = os.path.splitext(img_url)[1]
        #拼接图片名（包含路径）
        # filename = '{}{}{}{}'.format(file_path,os.sep,file_name,file_suffix)
        filename = '{}{}{}{}'.format(file_path,os.sep,file_name,'.jpg')
        # 下载图片，并保存到文件夹中
        request.urlretrieve(img_url,filename=filename)
    except IOError as e:
        print ('文件操作失败',e)
    except Exception as e:
        print ('错误 ：',e)

# 以下测试用
if __name__ == '__main__':
    File = open("sample_res.csv", "r")
    path_i = 0
    for line in File:
        try:
            img_url= line.split(' ')
            # 获取到id作为文件名字
            path_i = img_url[0]
            file_path = 'img/' + str(path_i)
            # 下面保存的是模版
            sample_img_url = img_url[1:-1]
            for s_img_url in sample_img_url:
                res_img_name = s_img_url.split('/')[-1].split('.')[:-1][0]
                save_img(s_img_url, res_img_name,file_path)

            # 下面保存的是作品图
            res_img_url = img_url[-1]
            res_img_name = res_img_url.split('/')[-1].split('.')[:-1][0]
            save_img(res_img_url, res_img_name,file_path)
            print (line)
        except:
            continue
    
    File.close()

    

