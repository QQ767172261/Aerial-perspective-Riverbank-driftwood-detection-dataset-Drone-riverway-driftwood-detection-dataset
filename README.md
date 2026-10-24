航拍视角河岸浮木检测数据集 无人机河道浮木检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4f711bad34b549ea801d8dbd19c624fe.png)
599张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ed52d68dead94e818a44dee788f1d980.png)

训练集图像数量:417; 验证集图像数量:120； 测试集图像数量:62
类别名称: 每一类图像数 ，每一类标注数
Riverbank-driftwood 河岸浮木 571 4966
image num: 599![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/319ba51807c443ceac13f4ca907c67f6.png)
# 航拍视角河岸浮木检测数据集
## 数据集信息总表
|项目|详情|
| ---- | ---- |
|数据集名称|航拍视角河岸浮木检测数据集|
|总图像数量|599张|
|图像分辨率|640×640|
|标注格式|YOLO、VOC、COCO三种|
|类别总数|1类|
|训练集|417张|
|验证集|120张|
|测试集|62张|

## 类别分布表
|类别英文|类别中文|包含该类图像数量|标注框总数|
| ---- | ---- | ---- | ---- |
|Riverbank-driftwood|河岸浮木|571|4966|

## 标签
`Riverbank-driftwood`

## 应用场景
1. 河道无人机巡检，识别洪水后河岸堆积浮木，评估河道堵塞风险；
2. 水文灾害监测，浮木堆积易阻塞河道、冲击堤坝，提前预警防洪隐患；
3. 流域生态环境监测，统计浮木分布，辅助河道清淤、清理作业规划；
4. 山洪灾害灾后评估，快速识别沿河漂浮物堆积区域。

## 数据集yaml配置 `river_driftwood.yaml`
```yaml
path: ./river_driftwood_dataset
train: images/train
val: images/val
test: images/test

names:
  0: Riverbank-driftwood
```

## YOLOv11训练代码 train_driftwood.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    result = model.train(
        data="river_driftwood.yaml",
        epochs=80,
        imgsz=640,
        batch=12,
        device=0,
        project="driftwood_output",
        name="yolo11_driftwood",
        conf=0.25,
        iou=0.45
    )
```

## 推理代码 predict_driftwood.py
```python
from ultralytics import YOLO

model = YOLO("./driftwood_output/yolo11_driftwood/weights/best.pt")

res = model.predict("test.jpg", save=True, conf=0.25)
for r in res:
    print(f"检测浮木数量：{len(r.boxes)}")
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]
        print(f"{cls_name}，置信度：{conf:.3f}，坐标：{xyxy}")
```

## 环境安装
```bash
pip install ultralytics torch opencv-python
```

关键词：航拍河岸浮木检测、河道漂浮物识别、无人机水文巡检、山洪灾害监测、河道堵塞预警、YOLO目标检测


