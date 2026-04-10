# RL from Scratch — N-in-a-Row with DQN

**強化学習をスクラッチで実装したプロジェクトです。**  
ニューラルネットワークからDQNエージェントまで、外部MLライブラリを使わずPythonのみで実装しました。

---

## 日本語

### プロジェクト概要

n×nの盤面で行う「n目並べ」を題材に、DQN（Deep Q-Network）エージェントをスクラッチで実装しました。
ニューラルネットワークの順伝播・逆伝播・パラメータ更新まで、NumPyやPyTorchなどのMLライブラリを一切使わずPythonリストで実装しています。

### ゲーム環境

- n×nの盤面をサポート
- 2人のプレイヤーが交互に手を打つ
- 行・列・対角線のいずれかを埋めたプレイヤーが勝利
- 盤面が全て埋まっても勝者がいない場合は引き分け
- 不正な手はexceptionを発生させる

**状態表現（DQNエージェント視点）**
- `1.0`：自分のマーク
- `0.0`：空きマス
- `-1.0`：相手のマーク

先手・後手どちらでも一貫した表現になります。

### DQNの設計

- ε-greedy探索
- オンラインネットワーク＋ターゲットネットワーク
- ベルマンターゲット
- デルタクリッピング
- 選択した行動のみQ値を更新

**ベルマンターゲット**

| 状態 | 式 |
|------|-----|
| 終端状態 | `y = r` |
| 非終端状態 | `y = r + γ * max_a' Q_target(s', a')` |

**報酬設計**

| 結果 | 報酬 |
|------|------|
| 勝利 | +1 |
| 敗北 | -1 |
| 引き分け | 0 |
| 途中ステップ | 0 |

### ニューラルネットワーク（スクラッチ実装）

| ファイル | 内容 |
|----------|------|
| `nn_activation.py` | Sigmoid / ReLU / パススルー |
| `nn_layer.py` | 線形変換・逆伝播・SGD/RMSProp |
| `nn_loss.py` | MSE / Huber損失 |
| `nn_matrix.py` | 行列・ベクトル演算 |
| `nn_mlp.py` | 順伝播・学習・JSONからの重み読み込み （未使用・将来対応）|

### ハイパーパラメータ（デフォルト値）

| パラメータ | 値 |
|-----------|-----|
| gamma | 0.95 |
| learning rate | 1e-3 |
| epsilon（初期） | 1.0 |
| epsilon（最終） | 0.1 |
| epsilon減衰ステップ | 10,000 |
| ターゲット更新間隔 | 500 |
| delta clip | 1.0 |
| optimizer | RMSProp |
| loss | MSE |
| 隠れ層 | (64, 64) |

### 実験結果（n=3、10万ゲーム）

| ルール | 引き分け | ランダム勝率 | DQN勝率 |
|--------|----------|--------------|---------|
| Rule 1（ランダム先手） | 0.050 | 0.222 | **0.729** |
| Rule 2（DQN先手） | 0.027 | 0.083 | **0.890** |
| Rule 3（勝者が後手） | 0.036 | 0.220 | **0.744** |

3つのルール全てでDQNエージェントがランダムエージェントを大幅に上回りました。

学習初期（〜2〜3万ゲーム）で勝率が急上昇し、その後は安定して推移しました。
評価が学習中に行われるため、ε-greedy探索の影響で勝率が若干ブレる点も確認しています。

### 今後の改善

- 学習と評価の分離
- 学習済みモデルの保存・読み込み
- 学習曲線の可視化
- 表形式Q学習との比較
- 人間対エージェントのインターフェース
- より大きな盤面サイズでの学習性能改善・検証

### 関連プロジェクト（作成予定）
このプロジェクトはDQNの実装に特化していますが、n目並べをより多角的に分析した関連プロジェクトを別途作成予定です。

局面の数学的分析 : brute-forceおよびバーンサイドの補題を用いた局面数の計算
Q学習の実装 : 表形式Q学習エージェントのスクラッチ実装
DQNとQ学習の比較 : n=3ではQ学習が有利な理由を含めた考察

### コードスタイルについて

実装経験が少ない段階でのプロジェクトのため、各ステップを明示化してミスを減らすことを目的に、意図的に冗長なコメントを残しています。

---

## English

### Overview

This project implements a DQN (Deep Q-Network) agent that learns to play N-in-a-Row (generalized Tic-Tac-Toe on an n×n board).  
The entire stack — from neural network operations to the DQN training loop — is built from scratch using only Python lists, without any ML libraries such as NumPy or PyTorch.

### Game Environment

- Supports n×n board sizes
- Two players alternate turns
- A player wins by filling an entire row, column, or diagonal
- If all cells are filled with no winner, the result is a draw
- Invalid actions raise an exception

**State Representation (from DQN agent's perspective)**
- `1.0`: my mark
- `0.0`: empty cell
- `-1.0`: opponent's mark

This encoding is consistent regardless of whether the DQN agent plays first or second.

### DQN Design

- ε-greedy exploration
- Online network + target network
- Bellman target
- Delta clipping
- Selected-action-only Q-value update

**Bellman Target**

| State | Formula |
|-------|---------|
| Terminal | `y = r` |
| Non-terminal | `y = r + γ * max_a' Q_target(s', a')` |

**Reward Design**

| Outcome | Reward |
|---------|--------|
| Win | +1 |
| Lose | -1 |
| Draw | 0 |
| Intermediate step | 0 |

### Neural Network (Scratch Implementation)

| File | Description |
|------|-------------|
| `nn_activation.py` | Sigmoid, ReLU, pass-through |
| `nn_layer.py` | Linear transform, backprop, SGD/RMSProp |
| `nn_loss.py` | MSE, Huber loss |
| `nn_matrix.py` | Matrix/vector operations |
| `nn_mlp.py` | Forward pass, training, JSON weight loading |

### Hyperparameters (Defaults)

| Parameter | Value |
|-----------|-------|
| gamma | 0.95 |
| learning rate | 1e-3 |
| epsilon start | 1.0 |
| epsilon end | 0.1 |
| epsilon decay steps | 10,000 |
| target update interval | 500 |
| delta clip | 1.0 |
| optimizer | RMSProp |
| loss | MSE |
| hidden layers | (64, 64) |

### Results (n=3, 100,000 games)

| Rule | Draw | Random Win | DQN Win |
|------|------|------------|---------|
| Rule 1 (Random first) | 0.050 | 0.222 | **0.729** |
| Rule 2 (DQN first) | 0.027 | 0.083 | **0.890** |
| Rule 3 (Winner plays second) | 0.036 | 0.220 | **0.744** |

The DQN agent significantly outperformed the random agent under all three rules.

Win rate improved sharply in the early stage (up to ~20,000–30,000 games), then stabilized with slight fluctuation — likely due to ε-greedy exploration continuing during evaluation.

### Future Improvements

- Separate training and evaluation phases
- Save and reload learned model parameters
- Visualize learning curves
- Compare DQN with tabular Q-learning
- Add a human-vs-agent interface
- Improve support for larger board sizes

### Motivation

This project was created to understand reinforcement learning more deeply by implementing the internal logic directly, rather than relying on high-level libraries.

### A Note on Code Style

Since this was an early implementation project, comments were written verbosely by design — to make each step explicit and minimize implementation errors.

---

## Project Structure

```
rl_from_scratch/
├── nn/
│   ├── nn_activation.py
│   ├── nn_layer.py
│   ├── nn_loss.py
│   ├── nn_matrix.py
│   └── nn_mlp.py
├── agent/
│   ├── agent_base.py
│   ├── random_agent.py
│   └── dqn_agent.py
├── env/
│   └── tictactoe.py
└── playgame_random_vs_dqn.py
```
