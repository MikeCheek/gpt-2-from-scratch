# GPT 2 LLM Architecture

The notebook of this repo contains code for building a GPT 2 language model from scratch, following the tutorial videos at this [this link](https://www.youtube.com/watch?v=yAcWnfsZhzo)

The code is available also on my Kaggle [notebook](https://www.kaggle.com/code/michelepulvirenti/gpt-llm)

## Code Structure

This implementation is organized into the following parts:

### 1. Setup and Configuration

- Environment setup and imports
- Model configurations for GPT-2 variants (Small, Medium, Large, XL)
- Device selection and tokenizer initialization

### 2. Core Architecture Modules

- **MultiHeadAttention**: Causal self-attention with masking
- **GELU & LayerNorm**: Activation and normalization layers
- **FeedForward**: Two-layer network with GELU activation

### 3. Model Structure

- **TransformerBlock**: Combines attention and feedforward with residual connections
- **GPTModel**: Token/positional embeddings, transformer stack, and output head

### 4. Text Generation

- `generate_text`: Next-token prediction with temperature and top-k sampling
- Tokenization utilities for string conversion

### 5. Transfer Learning

- TensorFlow checkpoint loading from OpenAI
- Weight mapping and conversion to PyTorch

### 6. Fine-tuning Pipeline

- Instruction formatting and data splitting
- `InstructionDataset` with dynamic padding and masking
- Custom collation for training

### 7. Training and Evaluation

- Loss calculation across batches and dataloaders
- Training loop with periodic evaluation and sampling
- Loss visualization

### 8. Inference and Testing

- Sample test cases for instruction-following
- Response extraction and comparison
