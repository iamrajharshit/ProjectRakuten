# Introduction
Quantization is a technique used in machine learning to reduce the size and computational requirements of a model by representing its parameters and operations with lower-precision.

## Why Quantization came into picture?

![SoomthQuant](img\quant\diag\01_SmoothQuant.png)
Reference: [SmoothQuant:AccurateandEfficientPost-TrainingQuantizationforLargeLanguageModels](https://arxiv.org/pdf/2211.10438)

Now a days, Deeplearning architecture are becoming larger and larger. In 2023-2024 largest most used models seems to be around ~70B parameters.

- This creates a gap between largest hardware and largest models.

- As 7B model would need approx. 280GB just to make the model fit on the hardware.

- Cunsumer-type hardware such as NVIDA T4 GPUs have only 16GB of RAM.

## Model Compression
![Model Compression](img\quant\diag\00_Compression.jpg)

Therefore the challenge is to make these state-of-the-art models accessible through model compression.

### State-of-the-art methods for model compression
#### Pruning

![Pruning](img\quant\diag\02_Pruning.png)

Pruning simply consists of removing layers in a model that do not have much importance on the model's decisions. It consists of removing some layers bansed on some matrics like magnitudes of the weights.

- The challenge here is that it is complex to tune and requires re-training to regain any lost accuracy.
- The level of compression may not be as high as that achieved through quantization in some cases.

#### Knowledge Distillation

![Knowledge Distillation](img\quant\diag\03_Knowledge Transfer.png)

Here, we train a student model, which is the target-compressed model using the output from the teacher model in addition to the main loss term.

- The challenge in knowledge distillation lies in ensuring that we have sufficient computational resources to load the original (teacher) model and generate its predictions.
- These predictions are then passed to the student model, and during this process, the loss is computed based on the difference between the student’s output and the teacher’s output.
- This process can be computationally expensive due to the need to run both the teacher and student models simultaneously during training.

## Quantization
![qunat](img\quant\diag\04_Quantiaztion.png)

Quantization simply consists of repesenting model weights or activations in a low precision.

- Idea it to store the parameters of the model in lower percision.

- The challenge here, is to lower the quantization error.



## Linear Qunatization


### Applying Linear Quantization using Quanto




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


## Papers on Quantization Methods
- LLM.int8(): 8-bit Matrix Multiplication
 for Transformers at Scale [arxiv.org](https://arxiv.org/pdf/2208.07339)
    - Proposed a no-performance degradation 8-bit quanitzation method by decomposing the underlying maxtrix multiplication in two stages, the outlier part in float16 and the non-outlier part in int8.
- QLORA:Efficient Finetuning of Quantized LLMs [arxiv.org](https://arxiv.org/pdf/2305.14314)
    - Making LLMs more accessible by quantizing them in 4-bit percision and being able to fine-tune, low-rank adapters on top of the model.

- SmoothQuant [arxiv.org](https://arxiv.org/pdf/2211.10438)
    - Propsoed to per-calibrate the model so that the quantized model does not get affected by large activations caused by large models.


