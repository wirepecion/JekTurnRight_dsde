import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from model import FloodLSTM
from data_utils import CONFIG, EarlyStopping, process_data, create_tensors, DS
from pathlib import Path
from src.setting.config import PROCESSED_DIR

# DATA_PATH should be defined within train.py or passed as an argument
DATA_PATH = "data/processed/flood_training_data_csv/part-00000-dc12f89c-7065-4d09-a951-dec13b2938ce-c000.csv"


if __name__ == "__main__":
    print(">>> Phase 1: Training Started")
    
    # LOAD YOUR DATA HERE
    df_full = pd.read_csv(DATA_PATH)
    df_clean = process_data(df_full)
    (X_tr, y_tr), (X_te, y_te), p_weight, input_dim = create_tensors(df_clean)

    tr_loader = DataLoader(DS(X_tr, y_tr), batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    te_loader = DataLoader(DS(X_te, y_te), batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    model = FloodLSTM(input_dim, CONFIG["HIDDEN_DIM"], CONFIG["LAYERS"], CONFIG["DROPOUT"]).to(CONFIG["DEVICE"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WD"])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p_weight).to(CONFIG["DEVICE"]))
    stop = EarlyStopping(patience=CONFIG["PATIENCE"])

    for ep in range(CONFIG["EPOCHS"]):
        model.train()
        t_loss = 0
        for X, y in tr_loader:
            X, y = X.to(CONFIG["DEVICE"]), y.to(CONFIG["DEVICE"])
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, y in te_loader:
                X, y = X.to(CONFIG["DEVICE"]), y.to(CONFIG["DEVICE"])
                v_loss += crit(model(X), y).item()

        print(f"Epoch {ep+1:02d}: Train {t_loss/len(tr_loader):.4f} | Val {v_loss/len(te_loader):.4f}")
        stop(v_loss/len(te_loader), model)
        if stop.early_stop: break

    # Save
    config = {"input_dim": input_dim, "hidden_dim": CONFIG["HIDDEN_DIM"], "num_layers": CONFIG["LAYERS"], "dropout": CONFIG["DROPOUT"]}
    with open("config.json", "w") as f: json.dump(config, f)
    torch.save(model.state_dict(), "pytorch_model.bin")
    print("\u2705 Phase 1 Complete.")
