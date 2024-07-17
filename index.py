import pandas as pd

# Load the Excel file into a pandas dataframe
df = pd.read_excel('results.xlsx', sheet_name='Sheet2')

# Convert the first column to numeric data types
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')

# Define the ranges you're interested in
ranges = [(0, 90), (90, 150), (150, 210), (210, 270), (270, 300)]

# Iterate over the ranges and extract the data
for start, end in ranges:
    data = df[(df.iloc[:, 1] >= start) & (df.iloc[:, 1] < end)]
    minor_count = (data.iloc[:, 2] == 'MINOR').sum()
    print(f'{start}-{end}: {minor_count} Minor')
