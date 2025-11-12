"""
env_wrapper.py

date : 2025/11/10
"""
import torch

class YOLOEnv:
    def __init__(self, yolo_model):
        self.model = yolo_model

    def reset(self):
        # 返回初始状态（例如：初始检测图像）
        state = torch.randn(1, 256)  # 随便举例，你自己替换
        return state

    def step(self, action):
        # action 可以是改变 YOLO 的某个超参数，例如 NMS 阈值
        # 然后重新跑一次 YOLO 推理，并算 reward

        reward = -abs(action.item() - 0.5)  # 示例，你需要自己定义
        next_state = torch.randn(1, 256)
        done = False

        return next_state, reward, done
