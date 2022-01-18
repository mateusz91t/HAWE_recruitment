import pandas as pd
from functions_and_data_cleaning_dir.additional_functions import (
    moving_mean,
    prepare_data_frame,
)


df = pd.read_csv("df_task_final.csv")
df = prepare_data_frame(df)
moving_mean(df, window=3)
