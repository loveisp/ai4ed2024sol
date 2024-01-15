import pandas as pd
from pathlib import Path
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("input_filename", type=str, help="input filename")
args = parser.parse_args()

input_fn = Path(args.input_filename)
output_path = Path('./submissions')
output_path.mkdir(parents=True, exist_ok=True)
output_fn = output_path / 'TAL_SAQ6K_EN_prediction.json'

df_ans = pd.read_json(input_fn, lines=True)
ans_pats = [
    'The answer is:', 
]
df_ans_sub = df_ans[df_ans.answer.str.contains('|'.join(ans_pats), regex=True)]
d_sub = {}
for queId, ans in zip(df_ans_sub.queId, df_ans_sub.answer.str.split('|'.join(ans_pats), regex=True).map(lambda x: x[-1].strip())):
    d_sub[queId] = ans.rstrip('.')
    
with open(output_fn, 'w') as f:
    json.dump(d_sub, f)