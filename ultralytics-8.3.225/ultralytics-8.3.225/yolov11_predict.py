from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO("D:\pig\\runs\\train_rl\\best_rl_run8\weights\\best.pt")  #修改为训练好的路径
source = 'test.png' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)