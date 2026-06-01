"""Compare time-series models on synthetic ERCOT-style load data."""

import numpy as np
import pandas as pd

np.random.seed(42)


def _synthetic_ercot(n_days: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    values = 25000 + 4000 * np.sin(np.linspace(0, 4 * np.pi, n_days)) + np.random.normal(
        0, 300, n_days
    )
    return pd.DataFrame({"date": dates, "values": values, "ds": dates, "y": values, "unique_id": "ercot"})


def load_temperature_data() -> None:
    df = _synthetic_ercot()
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        nf = NeuralForecast(
            models=[NBEATS(input_size=12, h=6, max_steps=10)],
            freq="D",
        )
        nf.fit(df=df[["unique_id", "ds", "y"]])
        print(nf.predict().head())
    except ImportError as exc:
        print(f"NeuralForecast unavailable ({exc}); printed synthetic head instead")
        print(df.head())


def notebook_step_002() -> None:
    import torch

    print("Torch Version:", torch.__version__)
    try:
        import neuralforecast

        print("NeuralForecast Version:", neuralforecast.__version__)
    except ImportError:
        print("NeuralForecast not installed")


def load_data() -> pd.DataFrame:
    return _synthetic_ercot()


def initialize_nbeats_model() -> None:
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        NeuralForecast(models=[NBEATS(input_size=12, h=6, max_steps=5)], freq="D")
    except ImportError:
        pass


def notebook_step_007() -> None:
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import AutoARIMA

        NeuralForecast(models=[AutoARIMA()], freq="D")
    except ImportError:
        pass


def notebook_step_008() -> None:
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
        from neuralforecast.utils import AirPassengersDF

        nf = NeuralForecast(models=[NBEATS(input_size=24, h=12, max_steps=5)], freq="ME")
        nf.fit(df=AirPassengersDF)
        print(nf.predict().head())
    except ImportError as exc:
        print(f"Skipping AirPassengers demo: {exc}")


def load_real_bakken_oil_production_data() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame({"ds": dates, "y": 1000 + rng.normal(0, 50, len(dates)).cumsum()})
    print("Synthetic Bakken production sample:")
    print(df.tail())


def main() -> None:
    notebook_step_002()
    load_temperature_data()
    load_data()
    initialize_nbeats_model()
    notebook_step_007()
    notebook_step_008()
    load_real_bakken_oil_production_data()


if __name__ == "__main__":
    main()
