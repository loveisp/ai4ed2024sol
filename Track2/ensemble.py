import pandas as pd
from pathlib import Path
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str, help="input dir")
args = parser.parse_args()

input_dir = Path(args.input_dir)


dfs = []
for fn in sorted(Path(input_dir).glob('*.jsonl')):
    df_ans = pd.read_json(fn, lines=True)
    ans_pats = [
        'The answer is:', 
    ]
    df_ans_sub = df_ans[df_ans.answer.str.contains('|'.join(ans_pats), regex=True)]
    d_sub = {}
    for queId, ans in zip(df_ans_sub.queId, df_ans_sub.answer.str.split('|'.join(ans_pats), regex=True).map(lambda x: x[-1].strip().strip('.'))):
        d_sub[queId] = ans
    dfs.append(pd.Series(d_sub).rename(fn.name.split('_')[1]))
df = pd.concat(dfs, axis=1)
df_mode = df.mode(1, dropna=True)
s_majors = df_mode[df_mode.isnull().sum(1) == df.shape[1]-1][0]
d_sub = {}
for queId, row in pd.concat([df['MetaMath-70B-V1.0'], s_majors], axis=1).iterrows():
    if pd.notnull(row[0]):
        d_sub[queId] = row[0]
    elif pd.notnull(row['MetaMath-70B-V1.0']):
        d_sub[queId] = row['MetaMath-70B-V1.0']
sub_root = Path('./submissions/')
sub_root.mkdir(parents=True, exist_ok=True)
sub_fn = sub_root / 'TAL_SAQ6K_EN_prediction.json'
with open(sub_fn, 'w') as f:
    json.dump(d_sub, f)