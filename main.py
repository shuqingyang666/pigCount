# main.py
import torch
from ultralytics import YOLO
from rl.actor_critic import ActorCritic
from rl.env_wrapper import YOLOEnv
import os

DATA_YAML = "D:/pig/pig_dataset6.yaml"  # ✅ YAML 数据集路径
MODEL_WEIGHTS = "yolo11n.pt"
PROJECT_DIR = "runs/train_rl"
RUN_NAME = "best_rl_run"

def decode_action(action):
    """将 RL 输出动作解码为 YOLO 参数"""
    conf = float(action[0]) * 0.5 + 0.25
    iou  = float(action[1]) * 0.5 + 0.30
    imgsz = int(round((640 + float(action[2]) * 256) / 32) * 32)
    return conf, iou, imgsz

def evaluate_policy(model, env, policy, num_eval_episodes=1):
    """评估策略平均奖励"""
    total_reward = 0.0
    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = policy.act(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state = next_state
        total_reward += ep_reward
    return total_reward / num_eval_episodes

def main():
    # 初始化 YOLO 模型与环境
    model = YOLO(MODEL_WEIGHTS)
    env = YOLOEnv(model, dataset_yaml=DATA_YAML)

    # 初始化 Actor-Critic 网络
    ac = ActorCritic(state_dim=256, action_dim=3)
    optim = torch.optim.Adam(ac.parameters(), lr=1e-4)

    num_episodes = 10
    gamma = 0.99
    best_reward = -float('inf')

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        transitions = []

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

            episode_rewards.append(reward)
            state = next_state

        total_reward = sum(episode_rewards)
        print(f"Episode {episode+1}/{num_episodes} — Total Reward: {total_reward:.4f}")

        # 每个 episode 结束后评估策略
        avg_reward = evaluate_policy(model, env, ac)
        print(f"Evaluation Reward: {avg_reward:.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            ac.save(os.path.join(PROJECT_DIR, "best_policy.pt"))
            print(f"✅ New best policy saved! Reward = {best_reward:.4f}")

    # =========================
    # 使用 RL 学得的最优参数训练 YOLO
    # =========================
    print("\n[RL → YOLO] Evaluate best policy for training parameters...")
    ac.load(os.path.join(PROJECT_DIR, "best_policy.pt"))
    state = env.reset()
    best_action, _ = ac.act(state)
    conf, iou, imgsz = decode_action(best_action)

    print(f"\n[RL Result] conf={conf:.2f}, iou={iou:.2f}, imgsz={imgsz}")

    model.train(
        data=DATA_YAML,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        epochs=150,
        batch=16,
        project=PROJECT_DIR,
        name=RUN_NAME
    )

    print("\n[YOLO Validation] Evaluating final model...")
    results = model.val(data=DATA_YAML, imgsz=imgsz)
    print(results.results_dict)

if __name__ == "__main__":
    main()
