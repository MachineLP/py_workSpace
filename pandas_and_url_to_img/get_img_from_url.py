# coding=utf-8
import pandas as pd  
import json
import os
from urllib import request
import json
import threading

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

def url_to_img(line):
    img_url= line.split(' ')
    # 获取到id作为文件名字
    path_i = img_url[0]
    print ('>>>>>', path_i)
    file_path = 'img/'  + str(path_i)
            
    # 判断是否已经保存模版，已经保存了就要避免重复保存。
    if not os.path.exists(file_path):
        # 下面保存的是模版
        sample_img_url = img_url[1]
        print ('>>>>>>', sample_img_url)
        res_img_name = sample_img_url.split('/')[-1].split('.')[:-1][0]
        save_img(sample_img_url, res_img_name,file_path)

    # 下面保存的是作品图
    res_img_url = img_url[2]
    res_img_url = res_img_url.split(',')
    for r_img_url in res_img_url:
        r_img_name = r_img_url.split('/')[-1].split('.')[:-1][0]
        save_img(r_img_url, r_img_name,file_path)
    print (res_img_url)

# 以下测试用
if __name__ == '__main__':
    File = open("material_url_v4.csv", "r")
    #print (len(File.readlines()))
    lines = File.readlines()
    print (lines)
    batch_size = 14
    for line_i in range(int(len(lines) / batch_size)):
        try:
            line_i_thread = lines[line_i*batch_size:(line_i+1)*batch_size]
            threads = [threading.Thread(target=url_to_img, args=(i_thread, )) for i_thread in line_i_thread]
            for t in threads:
                t.start()  #启动一个线程
            for t in threads:
                t.join()  #等待每个线程执行结束
        except:
            continue
    
    File.close()

'''
# 只拉模版
# 以下测试用
if __name__ == '__main__':
    File = open("sample_res.csv", "w")
    
    ############################################ 模版表 ##########################################
    # 这一部分对应的模版图
    pd_data_sample = pd.read_csv("模板表-material.csv")
    sample_img_url = pd_data_sample['preview 预览图：封面、1张预览、2张预览、3张预览']
    # print (sample_img_url.values)
    file_path = 'img/'
    for s_i_u in sample_img_url.values:
        try:
            print (s_i_u)
            sample_dic_jason = json.loads(str(s_i_u))
            print (sample_dic_jason)
            for sample_key in sample_dic_jason:
                s_img_url = sample_dic_jason[sample_key]
                print ('>>>',s_img_url)
                res_img_name = s_img_url.split('/')[-1].split('.')[:-1][0]
                save_img(s_img_url, res_img_name,file_path)
        except:
            continue
    File.close()

# 拉用户图和模版
'''

'''
# 以下测试用
if __name__ == '__main__':
    File = open("sample_res.csv", "r")
    for line in File:
        try:
            img_url= line.split(' ')
            # 获取到id作为文件名字
            path_i = img_url[0]
            file_path = 'img/'  + str(path_i)
            
            # 判断是否已经保存模版，已经保存了就要避免重复保存。
            if not os.path.exists(file_path):
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
'''

    



    

