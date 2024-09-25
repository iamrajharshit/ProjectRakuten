# Optimization of LLMs


## Workflow Architecture

![workflow](./img/home/00_Architecture.png)

1. The implementation of Particle Swarm Optimization (PSO) for weight extraction was initiated but could not be successfully completed.
    - [PSO Implementation](./pso.md) Doc.

2. Following a bottom-up approach, started working on the quantization of the model.
    - Quantized HuggingFace models, using Quanto Library.
    - Implemented 8-bit Modality Agnostic Quantizer.
    - Read [Quantization](./Quantization.md) Doc for more.

3. Implemented a character-wise GPT model to gain a deeper understanding of the GPT structure.
    - Read [GPT Implementation](./GPT.md) Doc for more.
    - Quantized the implemented GPT model.

4. Working on Finetuning the LLMs.
    - [Fine-tuning](./finetuning.md) Documentation.

## Future Work
- Learn and implement federated learning.
- Explore the implementation of a 2-bit quantizer for edge devices.
- Quantize and fine-tune the LLaMA model.


# [PSO Implementation ->](pso.md)





