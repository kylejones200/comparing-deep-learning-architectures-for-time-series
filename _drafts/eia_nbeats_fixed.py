import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from darts import TimeSeries
from darts.models import NBEATSModel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    input_chunk_length: int = 36
    output_chunk_length: int = 12
    epochs: int = 50


def load_series(cfg: Config) -> TimeSeries:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
    df = df.dropna().sort_values("date")
    ts = TimeSeries.from_dataframe(
        df, time_col="date", value_cols=["value"], freq=cfg.freq
    )
    return ts


def rolling_origin_nbeats(ts: TimeSeries, cfg: Config):
    s = ts.to_series()
    idx = np.arange(len(s))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    maes = []
    last_true, last_pred = None, None
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = ts.drop_after(ts.time_index[end])
        future = ts.split_after(ts.time_index[end])[1]
        y_te = future.drop_after(
            future.time_index[min(cfg.horizon - 1, len(future) - 1)]
        )
        if len(y_te) == 0:
            continue
        model = NBEATSModel(
            input_chunk_length=cfg.input_chunk_length,
            output_chunk_length=cfg.output_chunk_length,
            n_epochs=cfg.epochs,
            random_state=42,
            pl_trainer_kwargs={
                "enable_progress_bar": False,
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
            },
        )
        model.fit(y_tr)
        fc = model.predict(len(y_te))
        mae = mean_absolute_error(y_te.values().ravel(), fc.values().ravel())
        maes.append(mae)
        last_true, last_pred = y_te, fc
    return float(np.mean(maes)), (last_true, last_pred)


def main(plot: bool = False):
    cfg = Config()
    ts = load_series(cfg)
    mean_mae, (y_true, y_pred) = rolling_origin_nbeats(ts, cfg)
    logger.info(f"N-BEATS mean MAE: {mean_mae}")
    if plot:
        plt.figure(figsize=(9, 4))
        ts.plot(label="history", alpha=0.6)
        if y_pred is not None:
            y_pred.plot(label="N-BEATS last fold")
        plt.legend()
        signalplot.save("eia_nbeats_last_fold.png")


if __name__ == "__main__":
    main()
