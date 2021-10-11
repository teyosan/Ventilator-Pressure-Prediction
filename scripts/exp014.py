# Hide Warning
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Python Libraries
import sys
import os
import math
import random
import glob
import pickle
from collections import defaultdict
from pathlib import Path

# Third party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Utilities and Metrics
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_absolute_error #[roc_auc_score, accuracy_score]

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer, required
import torch_optimizer as optim

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

# Weights and Biases Tool
import wandb
#os.environ["WANDB_API_KEY"]='68fa1bbcda0fcf7a56f3c33a0fafa45b02f1c52d'
#wandb.login()


class CFG:
    debug = False
    competition = 'ventilator'
    exp_name = "exp014"
    seed = 29
    # model
    # img_size = 384

    # data
    target_col = 'pressure'  # 目標値のある列名
    target_size = 1

    # optimizer
    optimizer_name = 'RAdam'
    lr = 5e-3
    weight_decay = 1e-6
    amsgrad = False

    # scheduler
    epochs = 100
    scheduler = 'CosineAnnealingLR'
    T_max = 100
    min_lr = 1e-6
    # criterion
    # u_out = 1 を考慮しないLoss
    criterion_name = 'CustomLoss1'

    # training
    train = True
    inference = True
    n_fold = 5
    trn_fold = [0]
    precision = 16  # [16, 32, 64]
    grad_acc = 1
    # DataLoader
    loader = {
        "train": {
            "batch_size": 256,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True
        },
        "valid": {
            "batch_size": 256,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        }
    }
    # pl
    trainer = {
        'gpus': 1,
        'progress_bar_refresh_rate': 1,
        'benchmark': False,
        'deterministic': True,
    }
    # LSTM
    num_layers = 4
    cate_seq_cols = ['R', 'C']  # カテゴリー？
    cont_seq_cols = ['time_step', 'u_in', 'u_out'] + ['breath_time', 'u_in_time'] + ['u_in_cumsum', 'u_in_lag2']
    feature_cols = ['R', 'C', 'time_step', 'u_in', 'u_out', 'breath_time', 'u_in_time', 'u_in_cumsum', 'u_in_lag2']

    dense_dim = 512
    hidden_size = 512
    logit_dim = 512

INPUT_DIR = Path("F:/Kaggle/ventilator-pressure-prediction/data/input/")
OUTPUT_DIR = f'F:/Kaggle/ventilator-pressure-prediction/data/output/{CFG.exp_name}/'
df_train = pd.read_csv(INPUT_DIR / "train_v2.csv")
df_test = pd.read_csv(INPUT_DIR / "test_v2.csv")
submission = pd.read_csv(INPUT_DIR / "sample_submission.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if CFG.debug:
    CFG.epochs = 5
    CFG.inference = False
    df_train = df_train.head(240000)

#
# LINEに通知
import requests
def send_line_notification(message):
    line_token = '8vBbxd0jENU39kV2ROEwp78jAzeankBFi7AG0JjoU3j'
    endpoint = 'https://notify-api.line.me/api/notify'
    message = f"{message}"
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

# ==============================================================
# DataSet
# ==============================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        x = torch.FloatTensor(df[CFG.feature_cols].values)
        u_out = torch.LongTensor(df['u_out'].values)
        label = torch.FloatTensor(df['pressure'].values)
        return x, u_out, label


class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        x = torch.FloatTensor(df[CFG.feature_cols].values)
        return x

# ==============================================================
# DataModule
# ==============================================================
class DataModule(pl.LightningDataModule):

    def __init__(self, train_data, valid_data, test_data, cfg):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.cfg = cfg

    # 必ず呼び出される関数
    def setup(self, stage=None):
        self.train_dataset = TrainDataset(self.train_data)
        self.valid_dataset = TrainDataset(self.valid_data)
        self.test_dataset = TestDataset(self.test_data)

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.loader['train'])

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.cfg.loader['valid'])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.loader['valid'])


# ====================================================
# model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dense_dim = cfg.dense_dim
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.logit_dim = cfg.logit_dim
        self.mlp = nn.Sequential(
            nn.Linear(len(cfg.feature_cols), self.dense_dim // 2),
            nn.ReLU(),
            nn.Linear(self.dense_dim // 2, self.dense_dim),
            # nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.dense_dim, self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=0.2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.logit_dim),
            nn.ReLU(),
            nn.Linear(self.logit_dim, cfg.target_size),
        )
        # LSTMやGRUは直交行列に初期化する
        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f'init {m}')
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        bs = x.size(0)
        features = self.mlp(x)
        features, _ = self.lstm(features)
        output = self.head(features).view(bs, -1)
        return output


def get_model(cfg):
    model = CustomModel(cfg)
    return model


# ====================================================
# criterion
# ====================================================
def compute_metric(df, preds):
    """
    Metric for the problem, as I understood it.
    """

    y = np.array(df['pressure'].values.tolist())
    w = 1 - np.array(df['u_out'].values.tolist())

    assert y.shape == preds.shape and w.shape == y.shape, (y.shape, preds.shape, w.shape)

    mae = w * np.abs(y - preds)
    mae = mae.sum() / w.sum()

    return mae


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """

    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae


def get_criterion():
    if CFG.criterion_name == 'BCEWithLogitsLoss':
        # plだとto(device)いらない
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    if CFG.criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    if CFG.criterion_name == 'CustomLoss1':
        # [reference]https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
        criterion = VentilatorLoss()
    else:
        raise NotImplementedError
    return criterion


# ====================================================
# optimizer
# ====================================================
def get_optimizer(model: nn.Module, config: dict):
    """
    input:
    model:model
    config:optimizer_nameやlrが入ったものを渡す

    output:optimizer
    """
    optimizer_name = config.optimizer_name
    if 'Adam' == optimizer_name:
        return Adam(model.parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    amsgrad=config.amsgrad)
    elif 'RAdam' == optimizer_name:
        return optim.RAdam(model.parameters(),
                           lr=config.lr,
                           weight_decay=config.weight_decay)
    elif 'sgd' == optimizer_name:
        return SGD(model.parameters(),
                   lr=config.lr,
                   momentum=0.9,
                   nesterov=True,
                   weight_decay=config.weight_decay, )
    else:
        raise NotImplementedError


# ====================================================
# scheduler
# ====================================================
def get_scheduler(optimizer):
    if CFG.scheduler == 'ReduceLROnPlateau':
        """
        factor : 学習率の減衰率
        patience : 何ステップ向上しなければ減衰するかの値
        eps : nanとかInf回避用の微小数
        """
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True,
                                      eps=CFG.eps)
    elif CFG.scheduler == 'CosineAnnealingLR':
        """
        T_max : 1 半周期のステップサイズ
        eta_min : 最小学習率(極小値)
        """
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        """
        T_0 : 初期の繰りかえし回数
        T_mult : サイクルのスケール倍率
        """
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    else:
        raise NotImplementedError
    return scheduler


# ====================================================
# LightningModule
# ====================================================
class Trainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.criterion = get_criterion()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, u_out, y = batch
        # mixup とかしたい場合はここに差し込む
        output = self.forward(x)
        labels = y  # .unsqueeze(1)
        loss = self.criterion(output, labels, u_out).mean()
        # self.log_dict(dict(train_loss=loss))
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "labels": labels}

    def training_epoch_end(self, outputs):
        # training_stepの出力のまとまりがoutputsに入っている。
        self.log("lr", self.optimizer.param_groups[0]['lr'], prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, u_out, y = batch
        output = self.forward(x)
        labels = y  # .unsqueeze(1)
        loss = self.criterion(output, labels, u_out).mean()
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {"predictions": output,
                "labels": labels,
                "loss": loss.item()}

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        loss = 0
        for output in outputs:
            preds += output['predictions']
            labels += output['labels']
            loss += output['loss']

        labels = torch.stack(labels)
        preds = torch.stack(preds)
        loss = loss / len(outputs)

        self.log("val_loss_epoch", loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        output = self.forward(x)
        return output

    def test_step(self, batch, batch_idx):
        x = batch
        output = self.forward(x)
        return output

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self, self.cfg)
        self.scheduler = {'scheduler': get_scheduler(self.optimizer),
                          'interval': 'step',  # or 'epoch'
                          'frequency': 1}
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}


def train() -> None:
    seed_everything(CFG.seed)
    for fold in range(CFG.n_fold):
        if not fold in CFG.trn_fold:
            continue
        print(f"{'=' * 38} Fold: {fold} {'=' * 38}")
        # Logger
        # ======================================================
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # 学習済重みを保存するために必要
        loss_checkpoint = ModelCheckpoint(
            dirpath=OUTPUT_DIR,
            filename=f"best_loss_fold{fold}",
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
            save_weights_only=True,
            mode="min",
        )

        wandb_logger = WandbLogger(
            project=f'{CFG.competition}',
            group=f'{CFG.exp_name}',
            name=f'Fold{fold}',
            save_dir=OUTPUT_DIR
        )

        data_module = DataModule(
            df_train[df_train['fold'] != fold].reset_index(drop=True),
            df_train[df_train['fold'] == fold].reset_index(drop=True),
            df_test,
            CFG
        )
        data_module.setup()

        CFG.T_max = int(math.ceil(len(data_module.train_dataloader()) / CFG.grad_acc) * CFG.epochs)
        print(f"set schedular T_max {CFG.T_max}")
        # early_stopping_callback = EarlyStopping(monitor='val_loss_epoch',mode="min", patience=5)

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[loss_checkpoint],  # lr_monitor,early_stopping_callback
            default_root_dir=OUTPUT_DIR,
            accumulate_grad_batches=CFG.grad_acc,
            max_epochs=CFG.epochs,
            precision=CFG.precision,
            **CFG.trainer
        )
        # 学習
        model = Trainer(CFG)
        trainer.fit(model, data_module)
        torch.save(model.model.state_dict(), OUTPUT_DIR + '/' + f'{CFG.exp_name}_fold{fold}.pth')
        # best loss modelのロード
        best_model = Trainer.load_from_checkpoint(cfg=CFG, checkpoint_path=loss_checkpoint.best_model_path)
        # テストデータを予測して保存
        if CFG.inference:
            predictions = trainer.predict(best_model, data_module.test_dataloader())
            preds = []
            for p in predictions:
                preds += p
            preds = torch.stack(preds).flatten()
            submission['pressure'] = preds.to('cpu').detach().numpy()
            submission.to_csv(OUTPUT_DIR + '/' + f'submission_fold{fold}.csv', index=False)

        wandb.finish()

if __name__ == '__main__':
    train()
    send_line_notification("[locl]finished")
    wandb.finish()