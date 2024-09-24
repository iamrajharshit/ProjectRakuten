# Training the GPT model

## 290K Parameter BiG Language Model
- Configuration:
```
batch_size = 16    # no. of indep. seq. 
block_size = 32    # max context length for pred.
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64  # Embedding dimension 
n_head = 4   # Number of attention heads
n_layer = 4  # Number of transformer layers
dropout = 0.0
```
- Parameters:
```
0.209729 M parameters
```

- Training Metrics:
```
step 4999: train loss 1.6658, val loss 1.8275
```
- Output Generated (limited tokens):
```
ROTCUMER:
Tyburforth, bloody,
WhIs migute: you duke I use list. WIthon of where's grande will! savist tought!
Why room upwor alond, liegle. I hone, Iell thou sudd have then strue thus mind,
His by blow, Virdom tow, glingien, yithre spees ssince them Those not.

LUCIO:
Look,----
But thou sging them this my freceimmsed,
By thou sovor conursion that thou sade but grove
the tage encond:
It will Rament me; an your touther,
And havis like to-does, and little spright.
```
- Memory Footprint:
```
Model size: 0.86 MB
```

## 2.6M Parameter BiG Language Model
- Configuration:
```
batch_size = 16  # no. of indep. seq. 
block_size = 32  # max context length for pred.
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 192  # Embedding dimension 
n_head = 6    # Number of attention heads
n_layer = 6   # Number of transformer layers
dropout = 0.1

```

- Parameters:
```
2.697281 M parameters
```

- Training Metrics:
```
step 4999: train loss 1.5228, val loss 1.7088
```

- Output Generated (limited tokens):
```
CAPULET:
What thousal's sleep conceach!

MARIUS:
His now, where young that in buy gife,
And he deelinger if you, the treasmer owe lanch
As the make I can.
3reat his is a perforniced for sisson me made up,
Good to the love shalling free.
You busine England, bear his that wribuness, of his news so!
Stook you have ead, Bolinetant, say,
For a wombs, him less are throw hog, Upon his freat. Good-fear of noble.

```

- Memory Footprint:
```
Model size: 10.43 MB
```
## Papers On GPT
- Attention is All You Need [arxiv.org](https://arxiv.org/pdf/1706.03762)

# [Finetuning Page ->](finetuning.md)

## [<- Introduction to GPT](00_GPTintro.md)