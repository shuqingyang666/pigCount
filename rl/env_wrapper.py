import os
import torch
import yaml
from ultralytics import YOLO
from rl.map_reward import ap50_95  # 你之前的奖励函数

class YOLOEnv:
    def __init__(self, model, dataset_yaml):
        """
        model: YOLO 模型对象
        dataset_yaml: YAML 数据集配置文件
        """
        self.model = model

        # -------------------------
        # 解析 YAML 数据集路径
        # -------------------------
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)

        self.img_dir = data_cfg['train']
        self.lbl_dir = os.path.join(os.path.dirname(self.img_dir), "labels")  # 假设 labels 在同级 labels 文件夹

        # ✅ 加载合法图片
        self.images = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.images.sort()
        self.index = 0

        print(f"[INIT] Loaded {len(self.images)} images from {self.img_dir}")

    def reset(self):
        """重置环境"""
        self.index = 0
        return self._get_state()

    def step(self, action):
        """
        action: tensor(3)
        action[0] -> conf (0.25 – 0.75)
        action[1] -> iou  (0.30 – 0.80)
        action[2] -> imgsz (640 – 896)
        """
        action = action.squeeze()

        # -------- 1. 解码动作并限制范围 ----------
        conf = max(0.25, min(0.75, float(action[0]) * 0.5 + 0.25))
        iou  = max(0.30, min(0.80, float(action[1]) * 0.5 + 0.30))
        imgsz = int(round((640 + float(action[2]) * 256) / 32) * 32)

        self.model.overrides["conf"] = conf
        self.model.overrides["iou"] = iou
        self.model.overrides["imgsz"] = imgsz

        # -------- 2. 当前图像路径 ----------
        img_name = self.images[self.index]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] 图像不存在: {img_path}")

        # -------- 3. YOLO 推理 ----------
        try:
            result = self.model(img_path)[0]
        except Exception as e:
            print("[YOLO ERROR]", e)
            return self._get_state(), -1.0, True

        pred_boxes = result.boxes.xyxy.cpu()

        # -------- 4. 加载 GT ----------
        label_txt = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.lbl_dir, label_txt)
        gt_boxes = self._load_label(label_path)

        # -------- 5. 奖励计算 ----------
        try:
            out = ap50_95(pred_boxes, gt_boxes)
        except Exception as e:
            print("[REWARD ERROR]", e)
            return self._get_state(), -1.0, True

        if out is None:
            reward = -1.0
        else:
            ap50, map50_95 = out
            reward = ap50 + 2 * map50_95

        # -------- 6. 更新索引 ----------
        self.index += 1
        done = self.index >= len(self.images)

        next_state = self._get_state()
        return next_state, reward, done

    def _load_label(self, path):
        """YOLO txt → xyxy tensor"""
        if not os.path.exists(path):
            return torch.zeros((0,4))  # 空张量，避免 None

        boxes = []
        with open(path, 'r') as f:
            for line in f.readlines():
                items = line.strip().split()
                if len(items) != 5:
                    continue
                cls, x, y, w, h = map(float, items)
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                boxes.append([x1, y1, x2, y2])

        if boxes:
            return torch.tensor(boxes)
        else:
            return torch.zeros((0,4))

    def _get_state(self):
        """返回 RL 状态，可扩展为特征向量"""
        # 目前使用随机向量，可改为图像特征
        return torch.randn(1, 256)
