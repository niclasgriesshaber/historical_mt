# LLMs for Historical Machine Translation

In my master's dissertation, I conducted experiments with open-source LLMs on machine translation between English and early modern German language pairs on UCL's GPU cluster. I evaluated many-shot in-context learning, LoRA fine-tuning, and pre-trained instruct models on a privately held German legal dataset containing court protocols from the 16th to 18th century. The dataset includes almost 4,000 documents (1.4m tokens) that were transcribed and translated from archival sources by the Chichele Professor of Economic History, Sheilagh Ogilvie.

Much more research on historical, low-resource languages is needed to safely apply LLMs in historical research.

## Project Structure

- `data/` - Datasets, ICL examples, and prompts
- `scripts/` - Research notebooks and evaluation code
- `models/` - Fine-tuned model outputs
- `results/` - Evaluation results

## Quick Start

1. **Data preprocessing**: `scripts/data_preprocessing/`
2. **ICL experiments**: `scripts/[model]/icl/`
3. **Fine-tuning**: `scripts/[model]/finetuning/`
4. **Evaluation**: `scripts/[model]/evaluation/`

## Models

- **Meta**: Llama-3.1-8B-Instruct, Llama-3.2 variants
- **Google**: Gemma-2-9B-Instruct, Gemma-2-2B-Instruct
- **Microsoft**: Phi-3.5-mini-instruct

## Evaluation

- BLEU score
- COMET metric