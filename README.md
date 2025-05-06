---
license: cc-by-nc-4.0
datasets:
- Salesforce/xlam-function-calling-60k
language:
- en
pipeline_tag: text-generation
tags:
- function-calling
- LLM Agent
- tool-use
- llama
- qwen
- pytorch
- LLaMA-factory
library_name: transformers
---

<p align="center">
<img width="500px" alt="xLAM" src="https://huggingface.co/datasets/jianguozhang/logos/resolve/main/xlam-no-background.png">
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2504.03601">[Paper]</a>  |
  <a href="https://apigen-mt.github.io/">[Homepage]</a>  |
  <a href="https://huggingface.co/datasets/Salesforce/APIGen-MT-5k">[Dataset]</a> |
  <a href="https://github.com/SalesforceAIResearch/xLAM">[Github]</a>
</p>
<hr>

# Welcome to the xLAM-2 Model Family!

[Large Action Models (LAMs)](https://blog.salesforceairesearch.com/large-action-models/) are advanced language models designed to enhance decision-making by translating user intentions into executable actions. As the **brains of AI agents**, LAMs autonomously plan and execute tasks to achieve specific goals, making them invaluable for automating workflows across diverse domains.  
**This model release is for research purposes only.**  

The new **xLAM-2** series, built on our most advanced data synthesis, processing, and training pipelines, marks a significant leap in **multi-turn conversation** and **tool usage**. Trained using our novel APIGen-MT framework, which generates high-quality training data through simulated agent-human interactions. Our models achieve state-of-the-art performance on **BFCL** and **τ-bench** benchmarks, outperforming frontier models like GPT-4o and Claude 3.5. Notably, even our smaller models demonstrate superior capabilities in multi-turn scenarios while maintaining exceptional consistency across trials.

We've also refined the **chat template** and **vLLM integration**, making it easier to build advanced AI agents. Compared to previous xLAM models, xLAM-2 offers superior performance and seamless deployment across applications.  

<p align="center">
<img width="100%" alt="Model Performance Overview" src="https://github.com/apigen-mt/apigen-mt.github.io/blob/main/img/model_board.png?raw=true">
<br>
<small><i>Comparative performance of larger xLAM-2-fc-r models (8B-70B, trained with APIGen-MT data) against state-of-the-art baselines on function-calling (BFCL v3, as of date 04/02/2025) and agentic (τ-bench) capabilities.</i></small>
</p>


## Table of Contents
- [Model Series](#model-series)
- [Usage](#usage)
  - [Basic Usage with Huggingface Chat Template](#basic-usage-with-huggingface-chat-template)
- [Benchmark Results](#benchmark-results)
- [License](#license)
- [Citation](#citation)

## Model Series

We provide a series of xLAMs in different sizes to cater to various applications, including those optimized for multi-turn conversation and tool usage:

| Model                        | # Total Params | Context Length | Release Date | Base Model      | Category                                    | Download Model                                                               | Download GGUF files |
|------------------------------|----------------|----------------|--------------|-----------------|---------------------------------------------|------------------------------------------------------------------------------|---------------------|
| Salesforce/Llama-xLAM-2-70b-fc-r | 70B            | 128k            | Mar. 26, 2025 | Llama 3.1/3.2   | Multi-turn Conversation, Tool-usage           | [🤗 Link](https://huggingface.co/Salesforce/Llama-xLAM-2-70b-fc-r)         |      NA               |
| Salesforce/Llama-xLAM-2-8b-fc-r      | 8B             | 128k            | Mar. 26, 2025 |Llama 3.1/3.2       | Multi-turn Conversation, Tool-usage           | [🤗 Link](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)              |   [🤗 Link](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r-gguf)                  |
| Salesforce/xLAM-2-32b-fc-r     | 32B            | 32k (max 128k)*            | Mar. 26, 2025 | Qwen 2.5        | Multi-turn Conversation, Tool-usage           | [🤗 Link](https://huggingface.co/Salesforce/xLAM-2-32b-fc-r)             |      NA               |
| Salesforce/xLAM-2-3b-fc-r      | 3B             | 32k (max 128k)*            | Mar. 26, 2025 | Qwen 2.5        | Multi-turn Conversation, Tool-usage           | [🤗 Link](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r)              |      [🤗 Link](https://huggingface.co/Salesforce/xLAM-2-3b-fc-r-gguf)               |
| Salesforce/xLAM-2-1b-fc-r      | 1B             | 32k (max 128k)*            | Mar. 26, 2025 | Qwen 2.5        | Multi-turn Conversation, Tool-usage, Lightweight | [🤗 Link](https://huggingface.co/Salesforce/xLAM-2-1b-fc-r)              |      [🤗 Link](https://huggingface.co/Salesforce/xLAM-2-1b-fc-r-gguf)               |

***Note:** The default context length for Qwen-2.5-based models is 32k, but you can use techniques like YaRN (Yet Another Recursive Network) to achieve maximum 128k context length. Please refer to [here](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct#processing-long-texts) for more details.

## Usage

### Framework versions

- Transformers 4.46.1 (or later)
- PyTorch 2.5.1+cu124 (or later)
- Datasets 3.1.0 (or later)
- Tokenizers 0.20.3 (or later)

### Basic Usage with Huggingface Chat Template

The new xLAM models are designed to work seamlessly with the Hugging Face Transformers library and utilize natural chat templates for an easy and intuitive conversational experience. Below are examples of how to use these models.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Salesforce/Llama-xLAM-2-3b-fc-r")
model = AutoModelForCausalLM.from_pretrained("Salesforce/Llama-xLAM-2-3b-fc-r", torch_dtype=torch.bfloat16, device_map="auto")

# Example conversation with a tool call
messages = [
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in London?"},
]

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    }
]

print("====== prompt after applying chat template ======")
print(tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False))

inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
input_ids_len = inputs["input_ids"].shape[-1] # Get the length of the input tokens
inputs = {k: v.to(model.device) for k, v in inputs.items()}
print("====== model response ======")
outputs = model.generate(**inputs, max_new_tokens=256)
generated_tokens = outputs[:, input_ids_len:] # Slice the output to get only the newly generated tokens
print(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
```

<!-- ### Using vLLM for Inference

The xLAM models can also be efficiently served using vLLM for high-throughput inference. Please refer to the vLLM documentation for detailed instructions on how to deploy and use these models. You can typically start the vLLM service with the model name:

```bash
vllm serve Salesforce/xLAM-2-3b-fc-r
```

And then interact with the model using your preferred method for querying a vLLM endpoint. -->



## Benchmark Results

### Berkeley Function-Calling Leaderboard (BFCL v3)
<p align="center">
<img width="80%" alt="BFCL Results" src="https://github.com/apigen-mt/apigen-mt.github.io/blob/main/img/bfcl-result.png?raw=true">
<br>
<small><i>Performance comparison of different models on BFCL leaderboard. The rank is based on the overall accuracy, which is a weighted average of different evaluation categories. "FC" stands for function-calling mode in contrast to using a customized "prompt" to extract the function calls.</i></small>
</p>

### τ-bench Benchmark

<p align="center">
<img width="80%" alt="Tau-bench Results" src="https://github.com/apigen-mt/apigen-mt.github.io/blob/main/img/taubench-result.png?raw=true">
<br>
<small><i>Success Rate (pass@1) on τ-bench benchmark averaged across at least 5 trials. Our xLAM-2-70b-fc-r model achieves an overall success rate of 56.2% on τ-bench, significantly outperforming the base Llama 3.1 70B Instruct model (38.2%) and other open-source models like DeepSeek v3 (40.6%). Notably, our best model even outperforms proprietary models such as GPT-4o (52.9%) and approaches the performance of more recent models like Claude 3.5 Sonnet (new) (60.1%).</i></small>
</p>

<p align="center">
<img width="80%" alt="Pass^k curves" src="https://github.com/apigen-mt/apigen-mt.github.io/blob/main/img/pass_k_curves_retail_airline.png?raw=true">
<br>
<small><i>Pass^k curves measuring the probability that all 5 independent trials succeed for a given task, averaged across all tasks for τ-retail (left) and τ-airline (right) domains. Higher values indicate better consistency of the models.</i></small>
</p>


## Ethical Considerations

This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people's lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP. 

### Model Licenses

For all Llama relevant models, please also follow corresponding Llama license and terms. Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

## Citation

If you use our model or dataset in your work, please cite our paper:

```bibtex
@article{prabhakar2025apigenmt,
  title={APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay},
  author={Prabhakar, Akshara and Liu, Zuxin and Yao, Weiran and Zhang, Jianguo and Zhu, Ming and Wang, Shiyu and Liu, Zhiwei and Awalgaonkar, Tulika and Chen, Haolin and Hoang, Thai and Niebles, Juan Carlos and Heinecke, Shelby and Wang, Huan and Savarese, Silvio and Xiong, Caiming},
  journal={arXiv preprint arXiv:2504.03601},
  year={2025}
}
```

Additionally, please check our other related works regarding xLAM and consider citing them as well:

```bibtex
@article{zhang2025actionstudio,
  title={ActionStudio: A Lightweight Framework for Data and Training of Action Models},
  author={Zhang, Jianguo and Hoang, Thai and Zhu, Ming and Liu, Zuxin and Wang, Shiyu and Awalgaonkar, Tulika and Prabhakar, Akshara and Chen, Haolin and Yao, Weiran and Liu, Zhiwei and others},
  journal={arXiv preprint arXiv:2503.22673},
  year={2025}
}
```

```bibtex
@article{zhang2024xlam,
  title={xLAM: A Family of Large Action Models to Empower AI Agent Systems},
  author={Zhang, Jianguo and Lan, Tian and Zhu, Ming and Liu, Zuxin and Hoang, Thai and Kokane, Shirley and Yao, Weiran and Tan, Juntao and Prabhakar, Akshara and Chen, Haolin and others},
  journal={arXiv preprint arXiv:2409.03215},
  year={2024}
}
```

```bibtex
@article{liu2024apigen,
  title={Apigen: Automated pipeline for generating verifiable and diverse function-calling datasets},
  author={Liu, Zuxin and Hoang, Thai and Zhang, Jianguo and Zhu, Ming and Lan, Tian and Tan, Juntao and Yao, Weiran and Liu, Zhiwei and Feng, Yihao and RN, Rithesh and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={54463--54482},
  year={2024}
}
```

```bibtex
@article{zhang2024agentohana,
  title={AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning},
  author={Zhang, Jianguo and Lan, Tian and Murthy, Rithesh and Liu, Zhiwei and Yao, Weiran and Tan, Juntao and Hoang, Thai and Yang, Liangwei and Feng, Yihao and Liu, Zuxin and others},
  journal={arXiv preprint arXiv:2402.15506},
  year={2024}
}
```

