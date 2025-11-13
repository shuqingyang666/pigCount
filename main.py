import torch
from ultralytics import YOLO
from rl.actor_critic import ActorCritic
from rl.env_wrapper import YOLOEnv
import os

# -----------------------------
# 配置路径
# -----------------------------
DATA_YAML = "D:/pig/pig_dataset6/pig_dataset6.yaml"  # 指向 YAML 数据集
MODEL_WEIGHTS = "yolo11n.pt"  # 初始 YOLO 权重
PROJECT_DIR = "runs/train_rl"
RUN_NAME = "best_rl_run"

# -----------------------------
# RL 参数解码函数
# -----------------------------
def decode_action(action):
    """将 RL 输出动作解码为 YOLO 参数"""
    conf = float(action[0]) * 0.5 + 0.25
    iou  = float(action[1]) * 0.5 + 0.30
    imgsz = int(round((640 + float(action[2]) * 256) / 32) * 32)
    return conf, iou, imgsz

# -----------------------------
# 轻量验证函数
# -----------------------------
def evaluate_rl_action(model, dataset_yaml, action):
    conf, iou, imgsz = decode_action(action)
    model.overrides["conf"] = conf
    model.overrides["iou"] = iou
    model.overrides["imgsz"] = imgsz

    # 使用 YAML 文件初始化环境
    env_eval = YOLOEnv(model, dataset_yaml)
    total_reward = 0
    count = 0

    for _ in range(len(env_eval.images)):
        state = torch.randn(1, 256)
        _, reward, done = env_eval.step(action)
        total_reward += reward
        count += 1
        if done:
            break

    avg_reward = total_reward / max(count, 1)
    return avg_reward

# -----------------------------
# 主函数
# -----------------------------
def main():
    # 1️⃣ 初始化 YOLO 模型
    model = YOLO(MODEL_WEIGHTS)

    # 2️⃣ 初始化 RL 环境和 ActorCritic
    env = YOLOEnv(model, dataset_yaml=DATA_YAML)
    ac = ActorCritic(state_dim=256, action_dim=3)
    optim = torch.optim.Adam(ac.parameters(), lr=1e-4)

    best_reward = -float('inf')
    best_action = None
    num_episodes = 1
    gamma = 0.99

    # -------------------------
    # 3️⃣ RL 训练循环
    # -------------------------
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action, logp = ac.act(state)
            next_state, reward, done = env.step(action)

            value = ac.evaluate(state)
            next_value = ac.evaluate(next_state)
            advantage = reward + gamma * next_value - value

            actor_loss = -logp * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            state = next_state

            if reward > best_reward:
                best_reward = reward
                best_action = action.detach().cpu().numpy().squeeze()

        # ✅ 动态验证当前最优 RL 动作
        avg_reward = evaluate_rl_action(model, DATA_YAML, best_action)
        print(f"Episode {episode+1}/{num_episodes} finished. Current avg_reward={avg_reward:.4f}, Best reward so far={best_reward:.4f}")

    # -------------------------
    # 4️⃣ 输出 RL 最优参数
    # -------------------------
    conf, iou, imgsz = decode_action(best_action)
    print("\n[RL Result] Best action parameters:")
    print(f"conf={conf:.2f}, iou={iou:.2f}, imgsz={imgsz}")

    # -------------------------
    # 5️⃣ 用 RL 最优参数训练 YOLO
    # -------------------------
    print("\n[YOLO Training] Start training with RL-optimal parameters...")
    model.train(
        data=DATA_YAML,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        epochs=1,
        batch=16,
        project=PROJECT_DIR,
        name=RUN_NAME
    )

    # -------------------------
    # 6️⃣ 验证最终 mAP
    # -------------------------
    print("\n[YOLO Validation] Evaluate on dataset...")
    results = model.val(
        data=DATA_YAML,
        imgsz=imgsz,
        conf=conf,
        iou=iou
    )
    print("\n[Final Metrics]")
    mAP50 = results.results_dict['mAP_50']
    mAP50_95 = results.results_dict['mAP_50_95']
    print("mAP50:", mAP50)
    print("mAP50-95:", mAP50_95)
    print(f"YOLO weights saved at: {os.path.join(PROJECT_DIR, RUN_NAME, 'best.pt')}")

if __name__ == "__main__":
    main()
