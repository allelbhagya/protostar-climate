# features of time, last year, last 2nd year, last 3rd year

from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ds = load_dataset("CMU-OSPO/protostar-climate-train-to-2018", "annual_indices")
df = ds["train"].to_pandas()

target = "Avg Temp"
df = df.sort_values(["ID", "Year"])

for k in (1, 2, 3):
    df[f"lag_{k}"] = df.groupby("ID")[target].shift(k)

features = ["lag_1", "lag_2", "lag_3"]
df = df.dropna(subset=features + [target])
cutoff_year = df["Year"].quantile(0.8)
train_df = df[df["Year"] <= cutoff_year]
test_df = df[df["Year"] > cutoff_year]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(train_df[features], train_df[target])
pred = model.predict(test_df[features])

print(r2_score(test_df[target], pred))