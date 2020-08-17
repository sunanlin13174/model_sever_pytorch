# model_sever_pytorch
这是利用flask框架部署pytorch模型，其实很简单，只要求你对前端的知识有一点点了解就行  
CNN_classfiy_01 是对分类模型的命令行式的部署，先运行sever,后运行clinet,然后从clinet的命令行传入图片路径，得到返回结果。  
CNN_classfiy_02 是对分类模型的web界面部署，它有前端的代码，只需要运行sever.py然后，点击网页，鼠标操作即可得到结果。    
yolov3_model_serve_01 是对Yolov3模型的命令行式部署，先运行sever,后运行client,输入图片，得到框的坐标值和类别及置信度。没啥太大用  
yolov3_model_serve_0 是我写的核心，支持调用摄像头实时预测，以及输入视频路径。先运行，sever，后运行client,输入came，则调用本地摄像头，    
输入视频路径则实时显示结果，在本地。  
欢迎star,留言与我交流，后期会部署姿态估计模型，面部检测模型，图像分割模型，行人重识别模型等等。有打天池比赛的大佬，带我一个呀  
