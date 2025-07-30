from modes_main import modes
import glob
import pandas as pd

#'sample_results' directory contains csv files with filenames made of f'{job_id}_{task_id}.csv

best_fids = {}

for ii, mode in enumerate(modes):
    print(f'Analyzing mode {mode}..., ii= {ii}')
    
    regex_filename = f'sample_results/*_{ii}.csv'
    files = glob.glob(regex_filename)
    if not files:
        print(f'No files found for mode {mode}.')
        continue
    if len(files) > 1:
        print(f'Warning: More than one file found for mode {mode}. Using the first one.')
    filename = files[0]

    print(f'Processing file: {filename}')
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not lines:
            print(f'File {filename} is empty.')
            continue
        df = pd.read_csv(filename, header=None)
        if df.empty:
            print(f'File {filename} is empty after reading into DataFrame.')
            continue
        df.columns = ['fid_0', 'fid_1', 'fid_2', 'fid_3', 'fid_4']

        #set nan to 0
        df.fillna(0, inplace=True)

        df_binary = df > 0.99
        for ind, col in enumerate(df_binary.columns):
            #multiply each column by its index to get a binary representation
            df_binary[col] = df_binary[col] * (2 ** (len(df_binary.columns) - ind - 1))

        best_binary = df_binary.sum(axis=1).max()
        print(f'Best binary fidelity for mode {mode}: {best_binary}')

        df_best_binary = df[df_binary.sum(axis=1) == best_binary]
        print(f'shape of df_best_binary: {df_best_binary.shape}')

        best_avg_fid = df_best_binary.mean(axis=1).idxmax()
        print(f'Best average fidelity index for mode {mode}: {best_avg_fid}')

        best_values = df_best_binary.loc[best_avg_fid].values
        print(f'Best average fidelity values for mode {mode}: {best_values}')

        # print(f'Best average fidelity for mode {mode}: {best_values}')

        best_fids[mode] = best_values
result_df = pd.DataFrame()

for mode, values in best_fids.items():
    first_failure = next((i for i, v in enumerate(values) if v < 0.99), 3)
    print(f'First failure for mode {mode}: {first_failure}')
    for ii, value in enumerate(values):
        result_df = pd.concat([result_df, pd.DataFrame({
            'state': [mode],
            'vertnum': [ii],
            'fidelity': [value],
            'first_failure': [first_failure]
        })], ignore_index=True)


#plot state_length vs score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'serif'
# sns.set_style("whitegrid")
# sns.set_context("talk")

#plot fidelity vs vertnum for each state
plt.figure(figsize=(15, 5))

sns.lineplot(data=result_df, x="vertnum", y="fidelity", hue="first_failure", units="state", estimator=None, palette=["green", "lightblue", "orange", "lightpink"][::-1], linewidth=4)
plt.xlabel("Size index $N$", fontsize=16)
plt.xticks(np.arange(0, 5, 1), fontsize=16)
plt.yticks(fontsize=16)
plt.axvline(x=1, color='lightgrey', linestyle='--')
plt.axvline(x=2, color='lightgrey', linestyle='--')
plt.axvline(x=3, color='lightgrey', linestyle='--')
plt.axvline(x=4, color='lightgrey', linestyle='--')
plt.axvline(x=5, color='lightgrey', linestyle='--')
plt.axvline(x=6, color='lightgrey', linestyle='--')
plt.axvline(x=7, color='lightgrey', linestyle='--')
plt.ylabel("Fidelity", fontsize=16)
plt.title("Fidelity vs Number of Vertices", fontsize=16)
plt.ylim(0, 1.1)
plt.xlim(0, 4.0)
plt.tight_layout()
#remove legend
plt.legend().remove()

#make light purple background from 1 to 3 vertices
plt.axvspan(-0.1, 2, color='lavender', alpha=0.5)
#write text 'Training data region' in the middle of the purple background
plt.text(1, 0.033, "Training Range", ha='center', va='center', fontsize=16, color='black')

#write text 'Extrapolation' in the middle of the white background
plt.text(3, 0.033, "Extrapolation", ha='center', va='center', fontsize=16, color='black')


name_dict = {
    "bellN": "2d Bell",
    "bellN3d": "3d Bell",
    "w": "W-State",
    "ghz": "GHZ",
    "majumdar": "MG",
    "spinhalf": "Spin 1/2",
    "motzkin": "Motzkin",
    "motzkinsmall": "Motzkin Small",
    "dicke2dhalf2": "Dicke 1",
    "dyck": "Dyck 2",
    "ghzghz": "GHZ x GHZ",
    "dyck246": "Dyck 1",
    "ww": "W x W",
    "dicke2d2vsrest": "Dicke 4",
    "dicke2d2vsrest2": "Dicke 2",
    "ghzw": "GHZ x W",
    "dicke2d3vsrest2": "Dicke 3",
    "ghz3dghz3d": "GHZ 3D x GHZ 3D",
    "ww": "W x W",
    "ghzw": "GHZ x W",
    "aklt": "AKLT",
    "dicke": "Dicke 4",

}



print(name_dict.keys())
print(f'unique keys in result_df: {result_df["state"].unique()}')
print(f'difference between name_dict and result_df: {set(result_df["state"].unique())- set(name_dict.keys()) }')
# write state name for every line next to its value at x=5
# offset_step = 0.025
offset_step = 0.04
offset_5 = offset_step * 2.5 + 0.02
for key, value in name_dict.items():
    sub_df = result_df[result_df["state"] == key]
    print(f'length of {key} sub_df: {len(sub_df)}')
    for index, row in sub_df.iterrows():
        if (row["vertnum"] == row["first_failure"]):
            state_name = row["state"]
            fidelity = row["fidelity"]
            if row['first_failure'] == 3:
                offset = offset_5
                offset_5 -= offset_step
            else:
                offset = 0

            if key == "dicke2d2vsrest2":
                offset += -0.03
            if key == "majumdar":
                offset += 0.02
            if key == "aklt":
                offset += -0.02
            if key == "w":
                offset += 0.02
            if key == "dyck246":
                offset += -0.01
            if key == "dyck":
                offset += 0.01

            statetext = name_dict.get(state_name, state_name)
            plt.text(row["vertnum"]+0.05, fidelity-offset, statetext, ha='left', va='center', fontsize=9)

            #star at the anchor point
            plt.plot(row["vertnum"], fidelity, marker='*', markersize=8, color='black')



#save plot
plt.savefig("llama_bestfids.png")
