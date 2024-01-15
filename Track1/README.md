# Solutions for Track 1

Use multiple models to obtain answers to questions in the dataset, and then perform ensemble.

The scores of answers from each model after submission are as follows:

| model_name          | prompt_type | max_tokens | score |
|---------------------|-------------|------------|-------|
| MetaMath-70B-V1.0   | cot         | 512        | 11.30 |
| MetaMath-70B-V1.0   | default     | 512        | 10.72 |
| MetaMath-Mistral-7B | default     | 2048       | 10.13 |
| MetaMath-Mistral-7B | default     | 512        | 9.64  |
| MetaMath-Llemma-7B  | default     | 512        | 8.34  |
| MetaMath-7B-V1.0    | default     | 512        | 6.32  |
| MetaMath-13B-V1.0   | default     | 512        | 5.92  |
| Arithmo-Mistral-7B  | cot         | 512        | 5.70  |

- prompt_type:
    - "cot" refers to using "step by step" in the prompt.
    - "default" refers to not using "step by step" in the prompt.
- max_tokens refers to the maximum number of tokens in the output. The larger this value, the less likely the answer will be truncated, but it also significantly increases the inference time.
- score refers to the accuracy score obtained after submitting on the Codabench platform.

Since the scores are relatively low in this case, there is no ensemble performed like in Track 2. The best score obtained is 11.3 from the MetaMath-70B-V1.0 model.

## Inference

Executing the following code will provide you with the intermediate results of the six models with their inference processes.

```
CUDA_VISIBLE_DEVICES=0,1 python infer.py MetaMath-70B-V1.0 ./TAL-SAQ7K-CN.jsonl --prompt_type cot --max_tokens 512
```

Note that you need to set the path to the location of the downloaded models in the inference code.

After executing these codes, intermediate result files will be generated in the "raw_results" directory.

This step is very time-consuming and will take several hours, depending on the GPU performance.

## Extract Submission

By executing the following code, you can generate the final result file for submission in the "submissions" directory.

```
python extract_submission.py ./raw_results/TAL-SAQ7K-CN_XXX.jsonl
```

Here, we have already placed the inference result in the "raw_results" folder. We just need to execute the following command:

```
python extract_submission.py ./raw_results/TAL-SAQ7K-CN_MetaMath-70B-V1.0_512_cot.jsonl
```

This will generate the submission file with a score of 11.30 in the "submissions" directory.
