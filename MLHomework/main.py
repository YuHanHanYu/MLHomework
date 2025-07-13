from torch.utils.data import DataLoader
from utils.dataset import load_data
from utils.metrics import evaluate_model
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.cnn_transformer_model import CNNTransformerModel
from utils.plot_results import plot_prediction
import numpy as np

def run(horizon):
    train_set, test_set, scaler_y = load_data('data/train_process.csv', 'data/test_process.csv', input_len=90, pred_len=horizon)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)  # 单个sample用于展示可视化

    models = [
        ("LSTM", LSTMModel(input_dim=12, output_dim=horizon)),
        ("Transformer", TransformerModel(input_dim=12, output_dim=horizon)),
        ("CNN+Transformer", CNNTransformerModel(input_dim=12, output_dim=horizon))
    ]

    for name, model in models:
        mses, maes = [], []
        preds_list, trues_list = [], []

        for _ in range(5):
            mse, mae, preds, trues = evaluate_model(model, train_loader, test_loader, scaler_y, return_preds=True)
            mses.append(mse)
            maes.append(mae)
            preds_list.append(preds)
            trues_list.append(trues)

        print(f"{name} [{horizon}天] - MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}, MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        plot_prediction(trues_list[-1], preds_list[-1], title=f'{name} Forecast ({horizon} days)', horizon=horizon)

if __name__ == "__main__":
    run(90)
    run(365)
