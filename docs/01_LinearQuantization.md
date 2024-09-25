# Linear Qunatization
![Linear Quan](./img/quant/diag/08_Linear%20Quantization.png)
![Linear De-Quantization](./img/quant/diag/07_Linear%20Quantization.png)

Linear quantization is a technique used to reduce the memory footprint and computational cost of large language models (LLMs). It involves converting the model's weights and activations from high-precision floating-point numbers to lower-precision integers.

## Applying Linear Quantization using Quanto
Performing 8-Bit Precision using `Quanto` library.

- Check out the complete implementation in the [notebook](https://github.com/iamrajharshit/ProjectRakuten/blob/main/Quantization/02_T5%20FLAN%20Linear-quantization.ipynb).

### T5-FLAN Model
We will use [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) from Hugging face.

- It is a non fine-tuned small language model.
- It has 80M Parameters model. 

- Imoprting the model:
```
import sentencepiece as spm
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
```

#### Before Quantization

- Model:
```
T5ForConditionalGeneration(
  (shared): Embedding(32128, 512)
  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 512)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=512, out_features=384, bias=False)
              (k): Linear(in_features=512, out_features=384, bias=False)
              (v): Linear(in_features=512, out_features=384, bias=False)
              (o): Linear(in_features=384, out_features=512, bias=False)
              (relative_attention_bias): Embedding(32, 6)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=512, out_features=1024, bias=False)
              (wi_1): Linear(in_features=512, out_features=1024, bias=False)
              (wo): Linear(in_features=1024, out_features=512, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
...

```

- Memory:
```
The model size is 0.307844608 GB
```
- Output:
```
input_text = "where is Delhi"
inputs= tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=10)
output
```
```
tensor([[    0, 10619,     1]])
```
```
# lets decode
print(tokenizer.decode(output[0],skip_special_tokens=True))
```
```
Delhi
```
#### After Quantization

- Quantization Model:
```
T5ForConditionalGeneration(
  (shared): Embedding(32128, 512)
  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 512)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): QLinear(in_features=512, out_features=384, bias=False)
              (k): QLinear(in_features=512, out_features=384, bias=False)
              (v): QLinear(in_features=512, out_features=384, bias=False)
              (o): QLinear(in_features=384, out_features=512, bias=False)
              (relative_attention_bias): Embedding(32, 6)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): QLinear(in_features=512, out_features=1024, bias=False)
              (wi_1): QLinear(in_features=512, out_features=1024, bias=False)
              (wo): QLinear(in_features=1024, out_features=512, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
```
- Momery:
```
The model size is 0.12682868 GB
```

- Output:
```
input_text = "where is Delhi"
inputs= tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=10)
```
```
tensor([[    0, 10619,     1]])
```
```
print(tokenizer.decode(output[0]))
```
```
<pad> Delhi</s>

```
# [Quantization Granularity ->](02_QuantizationGranularity.md)
## [<- Introduction To Quantization](Quantization.md)
