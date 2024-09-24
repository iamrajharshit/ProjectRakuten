

#### 2.6M param Model
- This model has 2.6 Million parameters.

- For complete code implementation refer the [notebook](https://github.com/iamrajharshit/ProjectRakuten/blob/main/Quantization/10_Quantizing%202.2M%20param%20Model.ipynb).


##### Before Quantization 
- Model:
```
BigramLanguageModel(
  (token_embedding_table): Embedding(65, 192)
  (position_embedding_table): Embedding(32, 192)
  (blocks): Sequential(
    (0): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): Linear(in_features=192, out_features=32, bias=False)
            (query): Linear(in_features=192, out_features=32, bias=False)
            (value): Linear(in_features=192, out_features=32, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): Linear(in_features=192, out_features=192, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): Linear(in_features=192, out_features=768, bias=True)
          (1): ReLU()
          (2): Linear(in_features=768, out_features=192, bias=True)
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (1): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): Linear(in_features=192, out_features=32, bias=False)
            (query): Linear(in_features=192, out_features=32, bias=False)
            (value): Linear(in_features=192, out_features=32, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): Linear(in_features=192, out_features=192, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): Linear(in_features=192, out_features=768, bias=True)
          (1): ReLU()
          (2): Linear(in_features=768, out_features=192, bias=True)
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (2): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): Linear(in_features=192, out_features=32, bias=False)
            (query): Linear(in_features=192, out_features=32, bias=False)
            (value): Linear(in_features=192, out_features=32, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): Linear(in_features=192, out_features=192, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): Linear(in_features=192, out_features=768, bias=True)
          (1): ReLU()
          (2): Linear(in_features=768, out_features=192, bias=True)
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (3): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): Linear(in_features=192, out_features=32, bias=False)
            (query): Linear(in_features=192, out_features=32, bias=False)
            (value): Linear(in_features=192, out_features=32, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): Linear(in_features=192, out_features=192, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): Linear(in_features=192, out_features=768, bias=True)
          (1): ReLU()
          (2): Linear(in_features=768, out_features=192, bias=True)
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (4): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): Linear(in_features=192, out_features=32, bias=False)
            (query): Linear(in_features=192, out_features=32, bias=False)
            (value): Linear(in_features=192, out_features=32, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): Linear(in_features=192, out_features=192, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): Linear(in_features=192, out_features=768, bias=True)
          (1): ReLU()
          (2): Linear(in_features=768, out_features=192, bias=True)
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (5): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): Linear(in_features=192, out_features=32, bias=False)
            (query): Linear(in_features=192, out_features=32, bias=False)
            (value): Linear(in_features=192, out_features=32, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): Linear(in_features=192, out_features=192, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): Linear(in_features=192, out_features=768, bias=True)
          (1): ReLU()
          (2): Linear(in_features=768, out_features=192, bias=True)
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
  )
  (ln_f): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=192, out_features=65, bias=True)
)
```

- Model size:
```
Model size: 10.43 MB
```
##### After Quantization

- Quantized Model:
```
BigramLanguageModel(
  (token_embedding_table): Embedding(65, 192)
  (position_embedding_table): Embedding(32, 192)
  (blocks): Sequential(
    (0): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): W8A16LinearLayer()
            (query): W8A16LinearLayer()
            (value): W8A16LinearLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): W8A16LinearLayer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): W8A16LinearLayer()
          (1): ReLU()
          (2): W8A16LinearLayer()
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (1): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): W8A16LinearLayer()
            (query): W8A16LinearLayer()
            (value): W8A16LinearLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): W8A16LinearLayer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): W8A16LinearLayer()
          (1): ReLU()
          (2): W8A16LinearLayer()
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (2): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): W8A16LinearLayer()
            (query): W8A16LinearLayer()
            (value): W8A16LinearLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): W8A16LinearLayer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): W8A16LinearLayer()
          (1): ReLU()
          (2): W8A16LinearLayer()
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (3): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): W8A16LinearLayer()
            (query): W8A16LinearLayer()
            (value): W8A16LinearLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): W8A16LinearLayer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): W8A16LinearLayer()
          (1): ReLU()
          (2): W8A16LinearLayer()
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (4): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): W8A16LinearLayer()
            (query): W8A16LinearLayer()
            (value): W8A16LinearLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): W8A16LinearLayer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): W8A16LinearLayer()
          (1): ReLU()
          (2): W8A16LinearLayer()
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
    (5): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-5): 6 x Head(
            (key): W8A16LinearLayer()
            (query): W8A16LinearLayer()
            (value): W8A16LinearLayer()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (proj): W8A16LinearLayer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffwd): FeedFoward(
        (net): Sequential(
          (0): W8A16LinearLayer()
          (1): ReLU()
          (2): W8A16LinearLayer()
          (3): Dropout(p=0.1, inplace=False)
        )
      )
      (ln1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    )
  )
  (ln_f): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=192, out_features=65, bias=True)
)

```

- Model Memory:
```
Model size: 2.88 MB
```


## Papers on Quantization Methods
- LLM.int8(): 8-bit Matrix Multiplication
 for Transformers at Scale @[arxiv.org](https://arxiv.org/pdf/2208.07339)
    - Proposed a no-performance degradation 8-bit quanitzation method by decomposing the underlying maxtrix multiplication in two stages, the outlier part in float16 and the non-outlier part in int8.
- QLORA:Efficient Finetuning of Quantized LLMs @[arxiv.org](https://arxiv.org/pdf/2305.14314)
    - Making LLMs more accessible by quantizing them in 4-bit percision and being able to fine-tune, low-rank adapters on top of the model.

- SmoothQuant @[arxiv.org](https://arxiv.org/pdf/2211.10438)
    - Propsoed to per-calibrate the model so that the quantized model does not get affected by large activations caused by large models.

## Open source methods
These methods are designed to make LLMs smaller and faster, while minimizing performance degradation.

- QuIP: 2-Bit Quantization of
 Large Language Models 
    - Research Paper @[arxiv.org](https://arxiv.org/abs/2307.13304)
- AQLM:Extreme Compression of Large Language Models via Additive Quantization  
    - Research Paper @[arxiv.org](https://arxiv.org/pdf/2401.06118) 
    - [GitHub Repository](https://github.com/vahe1994/AQLM)

# [Introduction To GPT Page ->](00_GPTintro.md)
## [<- Bigram Model Quantization](03_BigramQunatization.md)
