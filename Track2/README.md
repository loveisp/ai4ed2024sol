# Solution for Track 2

Use multiple models to obtain answers to questions in the dataset, and then perform ensemble.

The scores of answers from each model after submission are as follows:

| model_name          | prompt_type | max_tokens | score |
|---------------------|-------------|------------|-------|
| MetaMath-70B-V1.0   | cot         | 512        | 42.86 |
| MetaMath-70B-V1.0   | default     | 512        | 42.41 |
| MetaMath-Mistral-7B | cot         | 512        | 40.66 |
| MetaMath-Mistral-7B | default     | 512        | 40.27 |
| MetaMath-Llemma-7B  | default     | 512        | 37.85 |
| Arithmo-Mistral-7B  | cot         | 512        | 37.85 |
| WizardMath-70B-V1.0 | cot         | 1024       | 34.76 |
| WizardMath-70B-V1.0 | cot         | 512        | 30.37 |

- prompt_type:
    - "cot" refers to using "step by step" in the prompt.
    - "default" refers to not using "step by step" in the prompt.
- max_tokens refers to the maximum number of tokens in the output. The larger this value, the less likely the answer will be truncated, but it also significantly increases the inference time.
- score refers to the accuracy score obtained after submitting on the Codabench platform.

We performed ensemble using the answers from six of the models, and in the end, we obtained a score of 48.76.

## Inference

Executing the following code will provide you with the intermediate results of the six models with their inference processes.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer.py MetaMath-70B-V1.0 ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 512
CUDA_VISIBLE_DEVICES=0,1 python infer.py MetaMath-Mistral-7B ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 512
CUDA_VISIBLE_DEVICES=0,1 python infer.py MetaMath-Mistral-7B ./TAL-SAQ6K-EN.jsonl --prompt_type default --max_tokens 512
CUDA_VISIBLE_DEVICES=0,1 python infer.py MetaMath-Llemma-7B ./TAL-SAQ6K-EN.jsonl --prompt_type default --max_tokens 512
CUDA_VISIBLE_DEVICES=0,1 python infer_arithmo.py ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 512
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer.py WizardMath-70B-V1.0 ./TAL-SAQ6K-EN.jsonl --prompt_type cot --max_tokens 1024
```

Note that you need to set the path to the location of the downloaded models in the inference code "infer.py".

After executing these codes, intermediate result files will be generated in the "raw_results" directory.

This step is very time-consuming and may take around 15 to 20 hours or even more, depending on the GPU performance.

## Extract Submission

By executing the following code, you can generate the final result file for submission in the "submissions" directory.

```
python extract_submission.py ./raw_results/TAL-SAQ6K-EN_XXX.jsonl
```

## Ensemble

Here, we use the intermediate result files of the 6 models obtained during the inference phase for ensemble.

The idea behind ensemble is as follows: we take a vote among the results from the 6 models. For a given queID, we remove the null values from the results of the 6 models. If the remaining results are completely consistent, we select that consistent result as the final result. If the results are inconsistent and the result from MetaMath-70B-V1.0 (the highest-scoring single model) is not a null value, we use the result from MetaMath-70B-V1.0 as the final result.

```
python ensemble.py ./ensemble_source/
```

After executing the code, the file "TAL_SAQ6K_EN_prediction.json" will be generated in the "submissions" folder for submission.

You can copy the 6 result files generated from the inference process to the "ensemble_source" folder, which will serve as the input for the ensemble.py script. For demonstration purposes, the "ensemble_source" folder already contains the result files from the inference process. Therefore, you can directly execute the code above without performing the inference. The "TAL_SAQ6K_EN_prediction.json" file generated is the result file with a score of 48.76.
