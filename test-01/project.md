- predicting value="avg temp"

features:
- lag 1: previous year's avg temp
- lag 2: temp from 2 years ago
- lag 3: temp from 3 years ago

| ID | Year | Avg Temp | lag_1 | lag_2 | lag_3 |
|---|---:|---:|---:|---:|---:|
| A | 2010 | 61.2 | NaN | NaN | NaN |
| A | 2011 | 61.8 | 61.2 | NaN | NaN |
| A | 2012 | 62.1 | 61.8 | 61.2 | NaN |
| A | 2013 | 62.5 | 62.1 | 61.8 | 61.2 |
| A | 2014 | 63.0 | 62.5 | 62.1 | 61.8 |

- removing rows without all three lags: year 2013, 2014 left

```python
X_train = train_df[["lag_1", "lag_2", "lag_3"]]
y_train = train_df["Avg Temp"]

model.fit(X_train, y_train)
```