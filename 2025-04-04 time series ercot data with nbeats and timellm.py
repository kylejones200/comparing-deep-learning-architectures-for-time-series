"""Generated from Jupyter notebook: 2025-04-04 time series ercot data with nbeats and timellm

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, TimeLLM


def main():
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
    import neuralforecast
    import torch

    print("Torch Version:", torch.__version__)
    print("NeuralForecast Version:", neuralforecast.__version__)
    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    df = pd.read_csv("ercot_load_data.csv")
    df["ds"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"values": "y"})
    df["unique_id"] = "temperature"
    nf = NeuralForecast(models=[NBEATS(input_size=12, h=6, max_steps=100)], freq="D")
    from neuralforecast.models import AutoARIMA

    nf = NeuralForecast(models=[AutoARIMA()], freq="D")
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from neuralforecast.utils import AirPassengersDF

    nf = NeuralForecast(models=[NBEATS(input_size=24, h=12, max_steps=100)], freq="ME")
    nf.fit(df=AirPassengersDF)
    nf.predict()
    import pandas as pd
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    from sklearn.ensemble import IsolationForest

    df = pd.read_csv("data/north_datoka_filtered_data_smaller.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df.set_index("ds", inplace=True)
    model = IsolationForest(contamination=0.05)
    df["anomaly"] = model.fit_predict(df[["y"]])
    anomalies = df[df["anomaly"] == -1]
    anomaly_prompt = "Recent oil production anomalies were detected: "
    anomaly_prompt += ", ".join(
        (
            f"{row.y} barrels on {row.Index.strftime('%B %d')}"
            for row in anomalies.tail(5).itertuples()
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
    main()


if __name__ == "__main__":
    main()
