
---

# Unsloth Fine-Tuning LLM

This repository provides a setup for fine-tuning the Unsloth model on your custom dataset and integrating it with Ollama. There are two primary ways to run and fine-tune the model: locally using `main.py` or using a Google Colab notebook for ease of use.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Local Setup with `main.py`](#local-setup-with-mainpy)
  - [Google Colab Setup](#google-colab-setup)
- [Dataset Preparation](#dataset-preparation)
  - [Manual Dataset Creation](#manual-dataset-creation)
  - [Using Hugging Face Datasets](#using-hugging-face-datasets)
- [Fine-Tuning the Model](#fine-tuning-the-model)
- [Saving and Running the Model with Ollama](#saving-and-running-the-model-with-ollama)
- [Commands for Ollama Integration](#commands-for-ollama-integration)
- [License](#license)

## Prerequisites

- Python 3.7+
- Hugging Face Account (for uploading datasets and models)
- Ollama (for running the model after fine-tuning)
- Google Colab (for running the notebook)
- Pip packages: `transformers`, `datasets`, `torch`, `onnx`, `tensorflow` (depending on your setup)

## Setup

### Local Setup with `main.py`

To run the fine-tuning process locally, you can use the `main.py` script. 

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python main.py
   ```

   Follow the instructions in the script to fine-tune the model on your dataset.

### Google Colab Setup

For easier setup, you can use the provided Colab notebook (`fine_tune_model.ipynb`). This is ideal if you are working with memory constraints, as Colab provides a GPU-backed environment.

1. Open the provided Colab notebook:
   - [Link to Google Colab Notebook](<notebook_link>)

2. Follow the instructions in the notebook to:
   - Prepare your dataset.
   - Fine-tune the model on the dataset.

   The notebook will guide you through the process, including how to upload your dataset to Hugging Face, run the training, and save the model.

## Dataset Preparation

### Manual Dataset Creation

You can manually create a dataset for fine-tuning in the format required by the model. The dataset should be in JSON format, with each entry containing the input and output prompts for the model.

Example format:
```json
[
  {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
  {"input": "Who is the president of the United States?", "output": "The president of the United States is Joe Biden."}
]
```

Upload your dataset to Hugging Face using the instructions provided in the Colab notebook.

### Using Hugging Face Datasets

Alternatively, you can use pre-existing datasets from Hugging Face. Search for datasets on the [Hugging Face Datasets Hub](https://huggingface.co/datasets) and load them directly into the Colab notebook or your local environment.

```python
from datasets import load_dataset

dataset = load_dataset('dataset_name')
```

## Fine-Tuning the Model

Once your dataset is prepared, use either the Colab notebook or the local script (`main.py`) to fine-tune the model. After training, the model will be saved in the `model` directory, and it will include the necessary files for running or further integration.

## Saving and Running the Model with Ollama

After fine-tuning the model, you may want to run it with Ollama, a tool for deploying LLMs.

### Option 1: Run with Colab (for Memory Constraints)

Due to memory constraints, running the model in Colab might be more efficient. Follow the steps outlined in the Colab notebook to run and test the model.

### Option 2: Run Locally and Save for Ollama Integration

If you want to save the trained model for Ollama integration, use the following command to save the model in the `.gguf` format.

```python
model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
```

This will save the model in the `model` directory as `unsloth.F16.gguf`.

## Commands for Ollama Integration

1. Create a `Modelfile` in the `model` folder where the `.gguf` file is located. Open the file in your terminal editor or manually.

2. Inside the `Modelfile`, write the following instructions:

```bash
FROM ./unsloth.F16.gguf
SYSTEM You are an anime expert
```

3. In your terminal, navigate to the folder containing the `Modelfile` and run the following command to create and deploy the model with Ollama:

```bash
ollama create <model_name> -f <Model_file_path>
```

4. Once the model is created, you can run the model using Ollama with:

```bash
ollama run <model_name>
```

This will start the model, and you can begin interacting with it.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
