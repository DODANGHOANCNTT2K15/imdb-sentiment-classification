import pandas as pd
import numpy as np

df = pd.read_csv("IMDB_Dataset.csv")

print("show 10 row:")
print(df.head(10))  
print("-------------------------------")
print("show info dataset")
print(df.describe())
