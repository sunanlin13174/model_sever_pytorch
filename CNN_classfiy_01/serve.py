import torch
import cv2
import os
import torch.nn as nn
import torchvision
import resnet
import time
from flask import request,Flask
from PIL import Image
from torchvision import transforms

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

val_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])
name_dict = {i:name.strip() for i ,name in enumerate(open('../name_json'))}



def  inference(img_transform,class_dict =None,img_path=''):
    img = cv2.imread(img_path)
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    pre_img = img_transform(img).unsqueeze(0)
    pre_img = pre_img.to(device)
    out =  net(pre_img).detach().numpy()
    index = out.argmax()
    return class_dict[index]

app = Flask(__name__)
@app.route('/',methods = ['POST'])
def respose_result():
    start_time = time.time()
    received_file = request.files['file']
    img_name = received_file.filename
    if received_file:
        save_path = './images'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        local_path = os.path.join(save_path,img_name)
        received_file.save(local_path)
        now_time = time.time()-start_time
        print(now_time)
        out = inference(val_transforms,name_dict,local_path)
        end_time = time.time()-start_time
        print(end_time)
        return out
if __name__=='__main__':
    # print('测试模型预测\n')
    # out = inference(val_transforms,name_dict,'/home/sal/bee.jpg')
    # print(out)
    app.run('127.0.0.1',port=5000)
