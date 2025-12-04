"""Transforms Number of infected people to number of new cases per day."""

import os
import pandas as pd
import tqdm

INPUT_DATAFOLDER = "src/data/covid/synthetic"
OUTPUT_DATAFOLDER = "src/data/covid/synthetic_transformed"

if not os.path.exists(INPUT_DATAFOLDER):
    raise ValueError("Input data folder does not exist.")

if not os.path.exists(OUTPUT_DATAFOLDER):
    os.makedirs(OUTPUT_DATAFOLDER)

for root, dirs, files in os.walk(INPUT_DATAFOLDER):
    for dir in tqdm.tqdm(dirs):
        for file in os.listdir(os.path.join(INPUT_DATAFOLDER, dir)):
            if not file.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(INPUT_DATAFOLDER, dir, file))
            last_day_infected = df.at[0, "Infected"]
            last_day_recovered = df.at[0, "Recovered"]
            for i, row in df.iterrows():
                if i == 0:
                    continue
                pre_last_day_infected = last_day_infected
                pre_last_day_recovered = last_day_recovered
                last_day_infected = df.at[i, "Infected"]
                last_day_recovered = df.at[i, "Recovered"]
                df.at[i, "Infected"] = df.at[i, "Infected"] - pre_last_day_infected + df.at[i, "Recovered"] - pre_last_day_recovered
            if not os.path.exists(os.path.join(OUTPUT_DATAFOLDER, dir)):
                os.makedirs(os.path.join(OUTPUT_DATAFOLDER, dir))
            df.to_csv(os.path.join(OUTPUT_DATAFOLDER, dir, file), index=False)
