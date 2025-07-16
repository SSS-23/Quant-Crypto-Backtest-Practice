# tcn_microservice_demo.py

# --- 1. 模型定义与训练 ---
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation,
                              dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else 64
            layers.append(TCNBlock(in_ch, 64, kernel_size=3, dilation=dilation))
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(64, output_dim)

    def forward(self, x):  # x: [batch, features, time]
        x = self.network(x)
        x = x[:, :, -1]  # take last time step
        return self.head(x)

# --- 2. 模拟数据集构建 ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 生成模拟数据：1000 条样本，每条样本为过去 30 步，3 个特征，预测1个数值
np.random.seed(0)
X = np.random.randn(1000, 3, 30).astype(np.float32)
y = np.random.rand(1000, 1).astype(np.float32)
dataset = TimeSeriesDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型训练
model = TCN(input_dim=3, output_dim=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for xb, yb in dataloader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "tcn_model.pth")


# --- 3. FastAPI 封装 ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 输入数据格式定义
class ModelInput(BaseModel):
    features: list[list[float]]  # shape: [features, time]

# 模型载入
model = TCN(input_dim=3, output_dim=1)
model.load_state_dict(torch.load("tcn_model.pth"))
model.eval()

@app.post("/predict")
async def predict(data: ModelInput):
    x = torch.tensor(data.features, dtype=torch.float32).unsqueeze(0)  # shape: [1, C, T]
    with torch.no_grad():
        y_pred = model(x)
    return {"predicted_value": y_pred.item()}

# --- 4. 模拟实盘调用方式 ---
# 你可以通过 curl / Postman / Python requests 调用这个接口：
# POST http://127.0.0.1:8000/predict
# Body JSON:
# {
#     "features": [[...], [...], [...]]  // 3 features x 30 timesteps
# }

# 启动方式（终端执行）：
# uvicorn tcn_microservice_demo:app --reload
