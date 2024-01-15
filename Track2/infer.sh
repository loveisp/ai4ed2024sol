CUDA_VISIBLE_DEVICES=2,3 python infer.py MetaMath-70B-V1.0 ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 512
CUDA_VISIBLE_DEVICES=2,3 python infer.py MetaMath-Mistral-7B ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 512
CUDA_VISIBLE_DEVICES=2,3 python infer.py MetaMath-Mistral-7B ./TAL-SAQ6K-EN.jsonl --prompt_type default --max_tokens 512
CUDA_VISIBLE_DEVICES=2,3 python infer.py MetaMath-Llemma-7B ./TAL-SAQ6K-EN.jsonl --prompt_type default --max_tokens 512
CUDA_VISIBLE_DEVICES=2,3 python infer_arithmo.py ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 512
CUDA_VISIBLE_DEVICES=2,3 python infer.py WizardMath-70B-V1.0 ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 1024