# Solutions for Track 2

Use multiple models to obtain answers to questions in the dataset, and then perform ensemble.

The scores of answers from each model after submission are as follows:

| model_name          | prompt_type | max_tokens | score |
|---------------------|-------------|------------|-------|
| MetaMath-70B-V1.0   | cot         | 512        | 42.86 |
| MetaMath-70B-V1.0   | default     | 512        | 42.41 |
| MetaMath-Mistral-7B | cot         | 512        | 40.27 |
| MetaMath-Mistral-7B | default     | 512        | 40.66 |
| MetaMath-Llemma-7B  | cot         | 512        | 37.85 |
| Arithmo-Mistral-7B  | cot         | 512        | 37.85 |
| WizardMath-70B-V1.0 | cot         | 1024       | 34.76 |
| WizardMath-70B-V1.0 | cot         | 512        | 30.37 |

- prompt_type:
    - "cot" refers to using "step by step" in the prompt.
    - "default" refers to not using "step by step" in the prompt.
    - Please refer to the code for the specific prompts.
- max_tokens refers to the maximum number of tokens in the output. The larger this value, the less likely the answer will be truncated, but it also significantly increases the inference time.
- score refers to the accuracy score obtained after submitting on the Codabench platform.

