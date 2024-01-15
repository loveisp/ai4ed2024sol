from vllm import LLM, SamplingParams
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm.auto import tqdm
import torch


DEBUG = False


parser = argparse.ArgumentParser()
parser.add_argument("input_filename", type=str, help="input filename")
parser.add_argument("--prompt_type", type=str, default="default", help="cot or pot")
parser.add_argument("--max_tokens", type=int, default=512, help="max tokens of output")
args = parser.parse_args()

input_fn = Path(args.input_filename)
prompt_type = args.prompt_type
max_tokens = args.max_tokens


def get_sampling_params(temperature=0., top_p=1., max_tokens=2048, stop=[]):
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)


def get_llm(model_path):
    num_gpus = torch.cuda.device_count()
    return LLM(model=model_path, tensor_parallel_size=num_gpus)


model_name = 'Arithmo-Mistral-7B'
# set the model paths here
model_path = '/home/ubuntu/data/huggingface/models/math/akjindal53244/Arithmo-Mistral-7B/'
llm = get_llm(model_path)

if prompt_type == 'cot':
    problem_prompt = "Question: {instruction}\n\nAnswer:"
elif prompt_type == 'pot':
    problem_prompt = "Question: {instruction}. Write a Python program to solve this.\n\nAnswer:"
else:
    raise NotImplementedError

stop_tokens = []
temperature = 0.
top_p = 1.
sampling_params = get_sampling_params(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop_tokens)

output_path = Path('./raw_results')
output_path.mkdir(parents=True, exist_ok=True)
output_fn = output_path / '{}_{}_{}_{}.jsonl'.format(input_fn.stem, model_name, max_tokens, prompt_type)


if DEBUG:
    df = pd.read_json(input_fn, lines=True).iloc[:100]
else:
    df = pd.read_json(input_fn, lines=True)
bs = 16
for i in tqdm(df.index[::bs].tolist()):
    queIds = []
    prompts = []
    for _, row in df.iloc[i:i+bs].iterrows():
        queIds.append(row.queId)
        prompts.append(problem_prompt.format(instruction=row.problem))
    completions = llm.generate(prompts, sampling_params)
    answers = []
    for output in completions:
        answers.append(output.outputs[0].text)
    with open(output_fn, 'a') as f:
        for queId, answer in zip(queIds, answers):
            f.write(json.dumps({'queId': queId, 'answer': answer}) + "\n")