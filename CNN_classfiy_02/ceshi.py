# -*- coding: utf-8 -*-
# 导入常用的库
import time
import os
import cv2
import numpy as np
# 导入flask库
import resnet
import  torch.nn as nn
import torch
from flask import Flask, render_template, request, jsonify
# 导入加载模型的方法load_model和模型编译使用的优化器类Adam


# 实例化Flask对象
app = Flask(__name__)
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 调用pickle库的load方法，加载图像数据处理时需要减去的像素均值pixel_mean

net  = resnet.resnet50()
net.fc = nn.Sequential(
    nn.Linear(2048,2)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_dict = torch.load('../resnet_frist_5_0.9477124214172363.pth')
for k,v in list(load_dict.items()):
    if 'fc'  in k:
        print(k)
        k_1 = 'fc.0.'+k.split('.')[-1]
        load_dict[k_1] = load_dict[k]
        del load_dict[k]

net.load_state_dict(load_dict)
net.to(device)


# 定义函数classId_to_className，把种类索引转换为种类名称
def classId_to_className(classId):
    category_list = ['ant','bee']
    className = category_list[classId]
    return className


# 根据图片文件的路径获取图像矩阵
def get_imageNdarray(imageFilePath):
    image_ndarray = cv2.imread(imageFilePath)
    resized_image_ndarray = cv2.resize(image_ndarray,
                                       (32, 32),
                                       interpolation=cv2.INTER_AREA)
    return resized_image_ndarray


# 模型预测前必要的图像处理
def process_imageNdarray(image_ndarray):
    rgb_image_ndarray = image_ndarray[:, :, ::-1]
    image_ndarray_1 = rgb_image_ndarray / 255
    image_ndarray_2 = image_ndarray_1
    return image_ndarray_2


# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称
def predict_image(model, imageFilePath):
    image_ndarray = get_imageNdarray(imageFilePath)
    processed_image_ndarray = process_imageNdarray(image_ndarray)
    inputs = processed_image_ndarray[np.newaxis, ...]
    predict_Y = model.predict(inputs)[0]
    predict_y = np.argmax(predict_Y)
    predict_classId = predict_y
    predict_className = classId_to_className(predict_classId)
    print('对此图片路径 %s 的预测结果为 %s' % (imageFilePath, predict_className))
    return predict_className


# 访问首页时的调用函数
@app.route('/')
def index_page():
    return render_template('cnn_classfiy_02.html')


# 使用predict_image这个API服务时的调用函数
@app.route("/predict_image", methods=['POST'])
def anyname_you_like():
    startTime = time.time()
    # 解析接收到的图片
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # 保存接收的图片到指定文件夹
        received_dirPath = './images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('接收图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        # 对指定图片路径的图片做分类预测，并打印耗时，返回预测种类名称
        startTime = time.time()
        predict_className = predict_image(net, imageFilePath)
        usedTime = time.time() - startTime
        print('完成对接收图片的分类预测，总共耗时%.2f秒\n' % usedTime)
        return jsonify(predict_className=predict_className)


# 主函数
if __name__ == "__main__":

    app.run("127.0.0.1", port=5000)