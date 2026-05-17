"""Generated from Jupyter notebook: 2025-04-04 time series ercot data with nbeats and timellm

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, TimeLLM


def main():
    # Load temperature data
    df = pd.read_csv("ercot_load_data.csv")  # Replace with actual file path
    df["ds"] = pd.to_datetime(df["date"])
    df["city"] = "ercot"
    df = df.rename(columns={"values": "y", "city": "unique_id"})

    # Initialize hybrid model: Time-LLM + NBEATS
    nf = NeuralForecast(
        models=[
            TimeLLM(input_size=24, h=6, max_steps=200),
            NBEATS(input_size=24, h=6, max_steps=200),
        ],
        freq="D",  # Daily temperature data
    )

    # Train the models together
    nf.fit(df=df)

    # Generate forecasts
    forecast = nf.predict()
    print(forecast.head())


    # --- code cell ---

    import neuralforecast
    import torch

    print("Torch Version:", torch.__version__)
    print("NeuralForecast Version:", neuralforecast.__version__)


    # --- code cell ---

    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    # Load data
    df = pd.read_csv("ercot_load_data.csv")
    df["ds"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"values": "y"})
    df["unique_id"] = "temperature"  # Ensure unique_id column


    # --- code cell ---

    # Initialize NBEATS model
    nf = NeuralForecast(models=[NBEATS(input_size=12, h=6, max_steps=100)], freq="D")


    # --- code cell ---

    # !pip list | grep lightning  # Jupyter-only


    # --- code cell ---

    # !pip uninstall pytorch-lightning -y  # Jupyter-only
    # !pip install pytorch-lightning==2.1.2  # Jupyter-only


    # --- code cell ---

    from neuralforecast.models import AutoARIMA

    nf = NeuralForecast(models=[AutoARIMA()], freq="D")


    # --- code cell ---

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from neuralforecast.utils import AirPassengersDF

    nf = NeuralForecast(models=[NBEATS(input_size=24, h=12, max_steps=100)], freq="ME")

    nf.fit(df=AirPassengersDF)
    nf.predict()


    # --- code cell ---

    import pandas as pd
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    from sklearn.ensemble import IsolationForest

    # Load real Bakken oil production data
    df = pd.read_csv(
        "/Users/kylejonespatricia/Downloads/north_datoka_filtered_data_smaller.csv"
    )
    df["ds"] = pd.to_datetime(df["ds"])
    df.set_index("ds", inplace=True)

    # Train anomaly detection model
    model = IsolationForest(contamination=0.05)
    df["anomaly"] = model.fit_predict(df[["y"]])

    # Extract flagged anomalies
    anomalies = df[df["anomaly"] == -1]

    # Format anomaly report for LLM
    anomaly_prompt = "Recent oil production anomalies were detected: "
    anomaly_prompt += ", ".join(
        f"{row.y} barrels on {row.Index.strftime('%B %d')}"
        for row in anomalies.tail(5).itertuples()
    )
    anomaly_prompt += ". What could be the cause of these anomalies?"

    # Call LLM
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        openai_api_key="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",
    )
    response = llm([HumanMessage(content=anomaly_prompt)])

    print(response.content)


    # --- code cell ---

    # !pip install --upgrade langchain langchain-community openai  # Jupyter-only


if __name__ == "__main__":
    main()
