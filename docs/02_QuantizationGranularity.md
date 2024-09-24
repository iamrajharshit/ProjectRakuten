

## Quantization Granularity

**Quantization granularity** refers to the level of detail at which a continuous value is represented in a discrete form. In the context of deep learning, it determines how finely the weights and activations of a neural network are quantized.

### Types of Quantization Granularity

1. **Per-Tensor Quantization:**
    - A single scale and zero-point are used for all elements in a tensor.
    - **Simplest** but may not be optimal for tensors with diverse distributions.
2. **Per-Channel Quantization:**
    - A separate scale and zero-point are used for each channel of a tensor.
    - **More flexible** than per-tensor quantization, especially for tensors with diverse distributions along different channels.
3. **Per-Group Quantization:**
    - A group of elements within a tensor is quantized using a single scale and zero-point.
    - **Intermediate** between per-tensor and per-channel quantization, offering a balance between accuracy and memory efficiency.

## 8-Bit Quantizer  

- Will quantize any model in `8-bit precision`.

- This quantizer is modality agnostic, meaning we can apply it on any model like vision, audio, text and even multi model.

- Will use Per-Channel Linear Quantization.

- Will create a `W8A16LinearLayer` class to store 8-bit weights and scales.

- Will replace all `torch.nn.Linear` layers with `W8A16LinearLayer`

<!-- - Then will build a quantizer and quantize a model end to end.

- Last but not the least will test the naive absmax quantization on many scenario and study its impact. -->


### Applying 8-Bit Quantization

#### Using [Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono) model from hugging face.

- So this is a Language MOdel that has been fine-tuned in code.
- And it has only 350million parameters.
- Lets use transformers to load the model with tokenizer and get some generation.
- For complete code refer the [notebook](https://github.com/iamrajharshit/ProjectRakuten/blob/main/Quantization/08_Custom%20Quantizer.ipynb).

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(model_id,
                                    torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```
##### Before Quantization

- Parameters:
```
model.get_parameter
```
```
<bound method Module.get_parameter of CodeGenForCausalLM(
  (transformer): CodeGenModel(
    (wte): Embedding(51200, 1024)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-19): 20 x CodeGenBlock(
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): CodeGenAttention(
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
          (qkv_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (mlp): CodeGenMLP(
          (fc_in): Linear(in_features=1024, out_features=4096, bias=True)
          (fc_out): Linear(in_features=4096, out_features=1024, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1024, out_features=51200, bias=True)
)>
```
- Memory Footprint:
```
print("Footprint of the model in MBs: ", 
      model.get_memory_footprint()/1e+6)

```
```
Footprint of the model in MBs:  797.310976
```

- Text genereated:
```
# the text generation piple to generate text
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# lets seee, as it is a model trainned and fine tuend on code, lets ask the model to complete code.
print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))

```
```
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
[{'generated_text': 'def hello_world():\n    print("Hello World")\n\n# hello_world()\n\n# def hello_'}]

```
#### Quantizing
- Replace linear with target and quantize model, TargetClass -> W8A16LinearLayer
- We r not quant {lm_head} because the model is an autoregressive model.
- As it uses the output from the previous iteration to get the output of the next iteration.
```
replace_linear_with_target_and_quantize(model,
                                        W8A16LinearLayer, ["lm_head"])
```
##### After Quantization 
- Parameters:
```
pipe.model.get_parameter
```
```
<bound method Module.get_parameter of CodeGenForCausalLM(
  (transformer): CodeGenModel(
    (wte): Embedding(51200, 1024)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-19): 20 x CodeGenBlock(
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): CodeGenAttention(
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
          (qkv_proj): W8A16LinearLayer()
          (out_proj): W8A16LinearLayer()
        )
        (mlp): CodeGenMLP(
          (fc_in): W8A16LinearLayer()
          (fc_out): W8A16LinearLayer()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1024, out_features=51200, bias=True)
)>
```


- Memory Footprint:
```
print("Footprint of the quantized model in MBs: ", 
      pipe.model.get_memory_footprint()/1e+6)

```
```
Footprint of the quantized model in MBs:  546.021376
```


- Text Generated:
    - When asked to define a function hello_world:
    ```
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    def hello_world():
        print("Hello World")

    # hello_world()

    # def hello_
    ```
    - When asked to define a function which returns sun of 5 natural numbers:
        ```
        Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
        def sum_of_natural_numbers(5):
            """Return sum of all numbers from 1 to 5."""
            sum = 0
            for i in range(1, 6):
                sum += i
            return sum

        # print(sum_of_natural_numbers(5))
        ```

# [Bigram Language Model 8-Bit Quantization ->](03_BigramQunatization.md)
## [<- Linear Quantization](01_LinearQuantization.md)