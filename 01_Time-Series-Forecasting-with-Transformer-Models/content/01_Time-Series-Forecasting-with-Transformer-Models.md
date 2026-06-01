# Time Series Forecasting with Transformer Models in Python

Transformer models reshaped natural language processing. They now shape time series forecasting. Attention layers help models decide which parts of a sequence matter most. The question is simple. Do Transformers beat ARIMA or LSTM on real data?

I ran all three models on Oklahoma energy consumption from 1960 to 2023. The series spans sixty four years of annual demand. The last twenty percent forms the test set.

### Dataset and Setup

The data comes from the State Energy Data System. It provides a long, smooth annual series that favors simpler models. Neural networks use normalized values. ARIMA uses the raw units. A twenty year lookback window gives each model the same context. The forecast horizon is five years.

### ARIMA

ARIMA gives a fast baseline. The (2,1,2) model captures the main trend. It runs in a fraction of a second. It does not handle complex non-linear structure. It returns error in the original energy units, which helps interpret the result.

### LSTM

LSTM handles non-linear patterns. I used two layers with fifty hidden units each. Dropout helps with generalization. Training takes a couple of seconds. It learns the slow drift in the series and fits the test set well.

### Transformer

The Transformer uses multi-head attention across two encoder layers. It focuses on which years matter most for each prediction. Training is slower. It needs more data to shine. On this annual univariate series it does not beat the LSTM.

### Results

| Model       |        MAE |       RMSE | Training Time (s) |
| ----------- | ---------: | ---------: | ----------------: |
| ARIMA       | 236,037.21 | 276,797.36 |              0.02 |
| LSTM        |     0.0704 |     0.0755 |              2.14 |
| Transformer |     0.7471 |     0.7478 |              5.45 |

ARIMA errors are in physical units. LSTM and Transformer errors use the normalized series. The LSTM provides the best fit. The Transformer lands behind it and adds cost in training time.

### Visual Comparison

The bar chart in `transformer_comparison.png` compares accuracy and runtime across models. The line plot in `forecast_comparison.png` overlays the actual series with the forecasts. You can see the LSTM track the test period more closely than the Transformer.

### Key Insights

The LSTM gives the best accuracy on this dataset. ARIMA gives the fastest baseline and keeps results in energy units. The Transformer gives attention weights you can inspect. It does not deliver higher accuracy on this simple series but scales well when you move to richer and more frequent data.

A single annual series limits the strength of a Transformer. It thrives on complex temporal structure. LSTM handles this problem with less cost and more accuracy.

### When Each Model Makes Sense

ARIMA helps when speed, simplicity, and interpretability matter.
LSTM helps when the data holds non-linear patterns and you want strong accuracy with modest cost.
Transformers help when long-range structure matters and you work with large, multivariate, or high-frequency datasets.

### Conclusion

Transformers push time series forecasting forward. They excel when data volume rises and long-range structure grows more complex. On this Oklahoma series the LSTM wins on accuracy and speed. ARIMA remains a strong baseline in physical units. The Transformer adds interpretability through attention but trails in error.

The best model depends on your data, your constraints, and the value you place on speed, insight, or raw accuracy.
