"""
main_rl_train.py

date : 2025/11/10
"""
import torch
from yolov11 import YOLO               # Ultralytics 的 YOLO 类
from rl.actor_critic import ActorCritic
from rl.env_wrapper import YOLOEnv

def main():
    # 1. 加载 YOLOv11
    model = YOLO("yolov11n.pt")

    # 2. 创建环境
    env = YOLOEnv(model)

    # 假设状态维度 256 / 动作数量 10（就当例子）
    state_dim = 256
    action_dim = 10

    ac = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-4)

    # 3. RL 训练循环
    for episode in range(50):
        state = env.reset()

        for t in range(200):
            action, log_prob = ac.act(state)
            next_state, reward, done = env.step(action)

            # Critic 的值
            value, _ = ac.evaluate(state)
            next_value, _ = ac.evaluate(next_state)

            advantage = reward + 0.99 * next_value - value
            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage.pow(2)

            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if done:
                break

        print(f"Episode {episode} done.")

if __name__ == "__main__":
    main()
