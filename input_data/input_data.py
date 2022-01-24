import pandas as pd
import json

df = pd.read_csv('raw_input_data.csv', sep=';')

data = {'timestamp': list(df['Timestamp']),
        'wind_velocity_mean': list(df['Wind_speed_mean']),
        'L50eq': list(df['L50eq']),
        'chronogram': list(df['chronogram'])
        }

with open('input_data.json', 'w+') as f:
    json.dump(data, f, indent=4)