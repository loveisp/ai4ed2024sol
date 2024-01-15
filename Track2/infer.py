from vllm import LLM, SamplingParams
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm.auto import tqdm
import torch


DEBUG = False


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name")
parser.add_argument("input_filename", type=str, help="input filename")
parser.add_argument("--prompt_type", type=str, default="default", help="default or cot")
parser.add_argument("--max_tokens", type=int, default=512, help="max tokens of output")
args = parser.parse_args()

model_name = args.model_name
input_fn = Path(args.input_filename)
prompt_type = args.prompt_type
max_tokens = args.max_tokens


def get_sampling_params(temperature=0., top_p=1., max_tokens=2048, stop=[]):
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)


def get_llm(model_path):
    num_gpus = torch.cuda.device_count()
    return LLM(model=model_path, tensor_parallel_size=num_gpus)


d_model_paths = {
    'MetaMath-Mistral-7B': '/home/ubuntu/data/huggingface/models/math/meta-math/MetaMath-Mistral-7B/', 
    'MetaMath-Llemma-7B': '/home/ubuntu/data/huggingface/models/math/meta-math/MetaMath-Llemma-7B/', 
    'MetaMath-7B-V1.0': '/home/ubuntu/data/huggingface/models/math/meta-math/MetaMath-7B-V1.0/', 
    'MetaMath-13B-V1.0': '/home/ubuntu/data/huggingface/models/math/meta-math/MetaMath-13B-V1.0/', 
    'MetaMath-70B-V1.0': '/home/ubuntu/data/huggingface/models/math/meta-math/MetaMath-70B-V1.0/', 
    'WizardMath-70B-V1.0': '/home/ubuntu/data/huggingface/models/math/WizardLM/WizardMath-70B-V1.0/', 
}
llm = get_llm(d_model_paths[model_name])

if prompt_type == 'default':
    problem_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
elif prompt_type == 'cot':
    problem_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
else:
    raise NotImplementedError

stop_tokens = ["Instruction:", "Instruction", "Response:", "Response", '</s>']
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