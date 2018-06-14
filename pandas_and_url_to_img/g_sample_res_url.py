# coding=utf-8
import pandas as pd  
import json

dic ={}
def json_txt(dic_json):
    if isinstance(dic_json,dict): #判断是否是字典类型isinstance 返回True false     
        for key in dic_json:
            if isinstance(dic_json[key],dict):#如果dic_json[key]依旧是字典类型
                print("****key--：%s value--: %s"%(key,dic_json[key]))
                json_txt(dic_json[key])
                dic[key] = dic_json[key]
            else:
                print("****key--：%s value--: %s"%(key,dic_json[key]))
                dic[key] = dic_json[key]

# 以下测试用
if __name__ == '__main__':
    File = open("sample_res.csv", "w")

    ############################################ 模版表 ##########################################
    # 这一部分对应的模版图
    pd_data_sample = pd.read_csv("模板表-material.csv")  
    sample_img_id = pd_data_sample['aterial_id']
    sample_img_url = pd_data_sample['preview 预览图：封面、1张预览、2张预览、3张预览']
    # sample_img_id = json.loads(sample_img_id.to_json())
    # sample_img_url = json.loads(sample_img_url.to_json())
    print (sample_img_id.values)
    print (sample_img_url.values)


    ############################################ 作品表 ##########################################
    #  这一部分对应的是作品表
    pd_data_res = pd.read_csv('作品表-work.csv')[:100]
    # print (pd_data_res.pop('作品内容'))
    # 转为json的格式
    res_dic_json = json.loads(pd_data_res['作品内容'].to_json())
    for res_key in res_dic_json:
        try:
            res_url_id = res_dic_json[res_key]
            # 作品表的id
            res_img_id = json.loads(res_url_id)['material_id']
            if res_img_id == 0:
                continue
            # 作品表的URL
            res_img_url = json.loads(res_url_id)['picture']
            print (res_img_id)
        
            # 用于保存模版的url。
            # sample_dic_url_list = []
            # print ( '>>>>', sample_img_url.values[res_img_id-1] )
            # 减1的目的因为数组从0开始的。
            sample_dic_jason = json.loads(sample_img_url.values[list(sample_img_id.values).index(res_img_id)])
            # 保存id， 用于找到一个id的图片
            File.write(str(res_img_id) + ' ')
            for sample_key in sample_dic_jason:
                print ('>>>>>>>>>>',sample_dic_jason[sample_key])
                # sample_dic_url_list.append(sample_dic_jason[sample_key])
                File.write(str(sample_dic_jason[sample_key]) + ' ')
            File.write(str(res_img_url) + '\n' )
        except:
            continue
    # 生成每行包含多个模版和一个作品
    File.close()

    

