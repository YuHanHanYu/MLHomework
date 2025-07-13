import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, train_loader, test_loader, scaler_y, epochs=20, lr=1e-3, return_preds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 训练阶段
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 测试阶段
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y = y.numpy()
            preds.append(pred)
            trues.append(y)

    # 反标准化
    preds = scaler_y.inverse_transform(np.concatenate(preds))
    trues = scaler_y.inverse_transform(np.concatenate(trues))

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)

    if return_preds:
        return mse, mae, preds[0], trues[0]  # 用于绘图的第一个样本
    return mse, mae
