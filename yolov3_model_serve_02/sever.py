import torch
import flask
import numpy
import base64
import argparse
from flask import request,Flask,render_template,jsonify
from yolov3_model_serve_01.yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3_model_serve_01.yolov3.utils.datasets import *
from yolov3_model_serve_01.yolov3.utils.utils import *
from yolov3_model_serve_02.client import get_drawedImage
app = Flask(__name__)


app.jinja_env.auto_reload= True
app.config['TEMPLATES_AUTO_RELOAD']= True

class pre_process:
    def __init__(self,path,img_size = 416,half =  False):
        path = str(Path(path))
        files = []
        if os.path.isfile(path):
            files = [path]
        else:
            ValueError()
            print('图片/视频路径错误\n')
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]  # 判断是否是支持的图片格式
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]  # 判断是否是支持的视频格式
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files 总共要检测的数目
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        self.half = half  # half precision fp16 images
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # 如果迭代次数等于图片数目，就停止迭代
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]  # 得到第self.count张图片路径

        if self.video_flag[self.count]:  # 如果有视频的话
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # 迭代次数加一
            self.count += 1
            # Read image 读取图片
            img0 = cv2.imread(path)  # BGR HWC: (1080, 810, 3)
            assert img0 is not None, 'Image Not Found ' + path
            # image 1/2 data/samples/bus.jpg:
            # print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.img_size)  # img经过padding后的最小输入矩形图: (416, 320, 3)

        # cv2.imshow('Padded Image', img)
        # cv2.waitKey()

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB  HWC2CHW: (3, 416, 320)
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files

def load_classes(path):
    with open(path,'r') as f:
        names = f.read().split('\n')
        return list(filter(None,names))

def get_secondFloat(timestamp):
    secondFloat = ('%.4f'%(timestamp%1))[1:]
    return secondFloat

def get_timeString():
    now_timetamp = time.time()
    now_structTime = time.localtime(now_timetamp)
    timeString_patten = '%Y%m%d_%H%M%S'
    now_timeString_1 = time.strftime(timeString_patten,now_structTime)
    now_timeString_2 = get_secondFloat(now_timetamp)
    now_timeString = now_timeString_1+now_timeString_2
    return now_timeString






def get_detectResult(cfg,img_path,weights_path,class_names_txt,img_size = 416,con_thres = 0.5,nms_thres = 0.5,save_img = True):
    obj =[]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    model = Darknet(cfg,img_size)
    #####
    if weights_path.endswith('.pt'):
        model.load_state_dict(torch.load(weights_path,map_location=device)['model'])
    else:
        _ = load_darknet_weights(model,weights_path)
    #eval mode
    model.to(device).eval()
    #Half precision
    opt.half = opt.half and device.type!='cpu'

    if opt.half:
        model.half()
    img_loader = pre_process(img_path,img_size=img_size,half=opt.half)
    classes = load_classes(class_names_txt)
    colors = [[random.randint(0,255) for _ in range(3)] for _ in range(len(classes))]

    #run inference
    for path,img,im0,vid_cap in img_loader:
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred,_ = model(img)
        det = non_max_suppression(pred.float(),con_thres,nms_thres)[0]
        # print(det)
        if det!=None:
            det[:, :4] = scale_coords(img.shape[2:],det[:, :4],im0.shape).round()
            # print('%gx%g'%(img.shape[2:]),end='')
            for c in det[:,-1].unique():
                n = (det[:,-1]==c).sum()
                # if classes[int(c)]=='person':
                #     print('%g %ss'%(n,classes[int(c)]))

            for *xyxy,cls_conf,cls in det:
                # print(xyxy)
                #
                # print(cls)
                xmin = int(xyxy[0])
                ymin = int(xyxy[1])
                xmax = int(xyxy[2])
                ymax = int(xyxy[3])
                obj.append(((xmin,ymin,xmax,ymax),int(cls),cls_conf.item()))
            return obj
        else:
            print('no det')
            return 0

from urllib.parse import unquote
def get_dataDict(data):
    data_dict= {}
    for text in data.split('&'):
        key,value = text.split('=')
        value_1 = unquote(value)
        data_dict[key] = value_1
    return data_dict


@app.route('/')
def index():
    return render_template('_14_yolov3.html')


@app.route('/get_detectionResult',methods = ['POST'])
def respose_result():
    xyxys=[]
    clss=[]
    confs = []
    t0 = time.time()
    data_bytes = request.get_data()
    data = data_bytes.decode('utf-8')
    data_dict = get_dataDict(data)
    if 'image_base64_string' in data_dict:
        # 保存接收的图片到指定文件夹
        received_dirPath = '../resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        timeString = get_timeString()
        imageFileName = timeString + '.jpg'
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        try:
            image_base64_string = data_dict['image_base64_string']
            image_base64_bytes = image_base64_string.encode('utf-8')
            image_bytes = base64.b64decode(image_base64_bytes)
            with open(imageFilePath, 'wb') as file:
                file.write(image_bytes)
            print('接收图片文件保存到此路径：%s' % imageFilePath)
            usedTime = time.time() - t0
            print('接收图片并保存，总共耗时%.2f秒' % usedTime)
            # 通过图片路径读取图像数据，并对图像数据做目标检测
            startTime = time.time()

            with torch.no_grad():
                out = get_detectResult(opt.cfg,imageFilePath,
                           weights_path=opt.weights_path,
                                 class_names_txt=opt.class_name_txt,
                                 img_size=416,
                             con_thres=0.5,)#out-》list
                # print(out)
            if out!=0:
                for obj in out:
                    # print(1)
                    xyxys.append(obj[0])

                    clss.append(obj[1])
                    confs.append(obj[2])
                object_time = time.time()-startTime
                print('检测图片共花费:%.3f s'%object_time)
                # print(xyxys)
                # drawed_image = get_drawedImage(image, xyxys, clss, confs)
                # drawed_imageFileName = 'drawed_' + imageFileName
                # drawed_imageFilePath = os.path.join(received_dirPath, drawed_imageFileName)
                # drawed_image.save(drawed_imageFilePath)
                # 把目标检测结果转化为json格式的字符串
                json_dict = {
                    'box_list': xyxys,
                    'classId_list': clss,
                    'score_list': confs
                }

                return jsonify(**json_dict)
            else:
                print('no det')
                json_dict = {
                    'box_list': [[0,0,0,0]],
                    'classId_list': [0],
                    'score_list': [0]
                }
                return jsonify(**json_dict)
        except Exception as e:
               print(e)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='H:/sal/model_serve/yolov3.cfg', help="模型配置文件路径")
    parser.add_argument('--weights_path', type=str, default='H:/sal/model_serve/yolov3.weights', help='模型权重文件路径')
    # parser.add_argument('--img_path', type=str, default='/home/sal/kitten.jpg', help='需要进行检测的图片文件夹')
    parser.add_argument('--class_name_txt',default='H:/sal/model_serve/yolov3_model_serve_02/yolov3/data/coco.names',help='coco_names的路径')
    parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS阈值')



    parser.add_argument('--half', default=False, help='是否采用半精度FP16进行推理')

    opt = parser.parse_args()
    # print(opt)

    # with torch.no_grad():
    #     out_txt = yolo3_detect(opt.cfg,opt.img_path,
    #            weights_path=opt.weights_path,
    #                  class_names_txt=opt.class_name_txt,
    #                  img_size=416,
    #                  con_thres=0.1,
    #                  nms_thres=0.4)
    # print(out_txt)
    app.run('127.0.0.1',port=5000)
