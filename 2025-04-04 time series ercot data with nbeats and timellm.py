"""Generated from Jupyter notebook: 2025-04-04 time series ercot data with nbeats and timellm

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import neuralforecast
import pandas as pd
import torch
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, AutoARIMA, TimeLLM
from neuralforecast.utils import AirPassengersDF
from sklearn.ensemble import IsolationForest


def load_temperature_data() -> None:
    df = pd.read_csv("ercot_load_data.csv")
    df["ds"] = pd.to_datetime(df["date"])
    df["city"] = "ercot"
    df = df.rename(columns={"values": "y", "city": "unique_id"})
    nf = NeuralForecast(
        models=[
            TimeLLM(input_size=24, h=6, max_steps=200),
            NBEATS(input_size=24, h=6, max_steps=200),
        ],
        freq="D",
    )
    nf.fit(df=df)
    forecast = nf.predict()
    print(forecast.head())


def notebook_step_002() -> None:
    print("Torch Version:", torch.__version__)
    print("NeuralForecast Version:", neuralforecast.__version__)


def load_data() -> None:
    df = pd.read_csv("ercot_load_data.csv")
    df["ds"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"values": "y"})
    df["unique_id"] = "temperature"


def initialize_nbeats_model() -> None:
    NeuralForecast(models=[NBEATS(input_size=12, h=6, max_steps=100)], freq="D")


def notebook_step_007() -> None:
    NeuralForecast(models=[AutoARIMA()], freq="D")


def notebook_step_008() -> None:
    nf = NeuralForecast(models=[NBEATS(input_size=24, h=12, max_steps=100)], freq="ME")
    nf.fit(df=AirPassengersDF)
    nf.predict()


def load_real_bakken_oil_production_data() -> None:
    df = pd.read_csv("data/north_datoka_filtered_data_smaller.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df.set_index("ds", inplace=True)
    model = IsolationForest(contamination=0.05)
    df["anomaly"] = model.fit_predict(df[["y"]])
    anomalies = df[df["anomaly"] == -1]
    anomaly_prompt = "Recent oil production anomalies were detected: "
    anomaly_prompt += ", ".join(
        (
            f"{row['y']} barrels on {row.name.strftime('%B %d')}"
            for _, row in anomalies.tail(5).iterrows()
        )
    )
    anomaly_prompt += ". What could be the cause of these anomalies?"
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        openai_api_key="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",
    )
    response = llm([HumanMessage(content=anomaly_prompt)])
    print(response.content)


def main() -> None:
    load_temperature_data()
    notebook_step_002()
    load_data()
    initialize_nbeats_model()
    notebook_step_007()
    notebook_step_008()
    load_real_bakken_oil_production_data()


if __name__ == "__main__":
    main()
