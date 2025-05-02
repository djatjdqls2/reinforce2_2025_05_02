from src6.CNN import CNNActionValue 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=50000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # 상태는 uint8로 저장 (0~255)
        self.s = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.uint8)

        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.s[self.ptr] = (state * 255).astype(np.uint8)
        self.s_prime[self.ptr] = (next_state * 255).astype(np.uint8)

        self.a[self.ptr] = action
        self.r[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = self.s[idx].astype(np.float32) / 255.0
        next_states = self.s_prime[idx].astype(np.float32) / 255.0

        return (
            states,
            self.a[idx],
            self.r[idx],
            next_states,
            self.done[idx]
        )

# DQN 신경망 모델 정의
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, lr=0.0005, epsilon=1.0, epsilon_min=0.1, gamma=0.9,
                 batch_size=32, warmup_steps=1000, buffer_size=int(5e4), target_update_interval=10000):
        
        super(DQN, self).__init__()

        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        # CNN 네트워크 정의
        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        
        # 네트워크 이름 일치시키기
        self.target_network.load_state_dict(self.network.state_dict())
        
        # 옵티마이저 설정
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        # 버퍼 설정
        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)  # ReplayBuffer 사용
        self.device = torch.device('cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        # 상태 및 epsilon 값 설정
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e5

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a

    def learn(self):
        if self.buffer.size < self.batch_size:
            return {}  # 학습 불가 상태

        s, a, r, s_prime, terminated = self.buffer.sample(self.batch_size)

        # 상태, 행동, 보상 등 torch tensor로 변환
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s_prime = torch.tensor(s_prime, dtype=torch.float32).to(self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(self.device)

        # Target network에서 다음 상태의 Q-value 계산
        next_q = self.target_network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.push(*transition)  # ✅ 반드시 수정

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            # 네트워크 업데이트 시 파라미터 이름 일치시킴
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return result