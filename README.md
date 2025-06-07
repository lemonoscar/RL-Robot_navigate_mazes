# 🤖 Maze-Solver with Deep Reinforcement Learning

本项目实现了一个基于深度强化学习（DQN）和启发式搜索（A*）的迷宫求解智能体，目标是在任意生成的迷宫中自主找到从起点到终点的最优路径。项目包含环境构建、强化学习模型训练、路径搜索算法、以及可视化评估流程。

---

### 1. 克隆项目

```bash
git clone https://github.com/your-username/MazeSolver.git
cd MazeSolver
````

### 2. 安装依赖

建议使用 Python 3.8+ 和虚拟环境：

```bash
pip install -r requirements.txt
```

依赖包括：

* `torch`
* `numpy`
* `matplotlib`（可选可视化）
* `tqdm`（进度条）

---

## 🧠 项目亮点

### ✅ 强化学习智能体（`Robot` 类）

* 基于 DQN 的策略网络训练路径选择
* 自定义奖励机制，引导机器人更快、更稳定地到达终点
* 支持训练与测试分离，便于评估策略效果
* 在静态迷宫中构建完整状态空间进行训练（offline RL）

### 🔍 启发式路径搜索（`my_search`）

* 实现 A\* 算法用于对比与基准路径验证
* 启发函数采用 Manhattan 距离，适合离散格子迷宫




