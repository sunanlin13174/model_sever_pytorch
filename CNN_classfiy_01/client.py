import requests
import os
if __name__=='__main__':
    url = 'http://127.0.0.1:5000'
    while True:
        input_img = input('请输入要分类的图片路径：-1🖕中止\n')
        if input_img.strip()=='-1':
            break
        elif input_img.strip()=='':
            continue
        else:
            img_name = input_img.strip(' ').split('/')[-1]
            img_rb = open(input_img,'rb')
            files_dict = {
                'file':(img_name,img_rb,'image/jpg')
            }
            result_respose = requests.post(url,files = files_dict)
            print('在线模型预测结果为：%s'%result_respose.text)