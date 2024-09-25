# **Finetuning of LLMs**

<!-- ###  [*Glossary*](https://developers.google.com/machine-learning/glossary) -->

A *foundation LLM* is trained on enough natural language to "know" a remarkable amount about grammar, words, and idioms. A foundation language model can generate helpful sentences about topics it is trained on. Furthermore, a foundation LLM can perform certain tasks traditionally called "creative," like writing poetry. However, a foundation LLM's generative text output isn't a solution for other kinds of common ML problems, such as regression or classification. For these use cases, a foundation LLM can serve as a platform rather than a solution.

  Transforming a foundation LLM into a solution that meets an application's needs requires a process called fine-tuning. A secondary process called distillation generates a smaller (fewer parameters) version of the fine-tuned model.

## **Fine-tuning** 

Research shows that the pattern-recognition abilities of foundation language models are so powerful that they sometimes require relatively little additional training to learn specific tasks. That additional training helps the model make better predictions on a specific task. This additional training, called [*fine-tuning*](https://developers.google.com/machine-learning/glossary#fine-tuning) , unlocks an LLM's practical side.


Fine-tuning trains on examples specific to the task your application will perform. Engineers can sometimes fine-tune a foundation LLM on just a few hundred or a few thousand training examples.

Despite the relatively tiny number of training examples, standard fine-tuning is often computationally expensive. That's because standard fine-tuning involves updating the weight and bias of every parameter on each backpropagation iteration. Fortunately, a smarter process called parameter-efficient tuning can fine-tune an LLM by adjusting only a subset of parameters on each backpropagation iteration.

A fine-tuned model's predictions are usually better than the foundation LLM's predictions. However, a fine-tuned model contains the same number of parameters as the foundation LLM. So, if a foundation LLM contains ten billion parameters, then the fine-tuned version will also contain ten billion parameters.

## **Distillation**

Most fine-tuned LLMs contain enormous numbers of parameters. Consequently, foundation LLMs require enormous computational and environmental resources to generate predictions. Note that large swaths of those parameters are typically irrelevant for a specific application.

Distillation creates a smaller version of an LLM. The distilled LLM generates predictions much faster and requires fewer computational and environmental resources than the full LLM. However, the distilled model's predictions are generally not quite as good as the original LLM's predictions. Recall that LLMs with more parameters almost always generate better predictions than LLMs with fewer parameters.


Before Starting with Fine Tuning: 

Understand how better you can quantize the model
## [Compute cost and requirement](https://blog.eleuther.ai/transformer-math/)

# Code Implementation

``` python
"""
Base minimal code for finetuning smls, we can finetune desired smls with our custom data and analyze the prediction
"""

data_sample = load_dataset("databricks/databricks-dolly-15k")
print(data_sample)

# # Convert to a pandas dataframe
#updated_data = [{'Instruction': item['instruction'], 'Response': item['response']} for item in data_sample['train']]
df =pd.DataFrame(data_sample['train'])

# df.head(5)

df_combined = pd.concat([df['instruction'], df['response']], axis=1)
# # Just extract the Symptoms
# df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x.split(', ')))
# print(df.head())


# If you have an NVIDIA GPU attached, use 'cuda'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

# The tokenizer turns texts to numbers (and vice-versa)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# The transformer
model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)



print(model) #this display the model configuration such as layers, nodes etc

# Model params CONST
BATCH_SIZE = 8

df.describe()

# Cast the Huggingface data set as a LanguageDataset we defined above
data_sample = LanguageDataset(df_combined, tokenizer)


# Create train, valid
train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(data_sample, [train_size, valid_size])

# Make the iterators
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

# Set the number of epochs
num_epochs = 1

# Training parameters
batch_size = BATCH_SIZE
model_name = 'distilgpt2'
gpu = 0

# Set the learning rate and loss function
## CrossEntropyLoss measures how close answers to the truth.
## More punishing for high confidence wrong answers
criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
tokenizer.pad_token = tokenizer.eos_token

# Init a results dataframe
results = pd.DataFrame(columns=['epoch', 'transformer', 'batch_size', 'gpu',
                                'training_loss', 'validation_loss', 'epoch_duration_sec'])

# The training loop
for epoch in range(num_epochs):
    start_time = time.time()  # Start the timer for the epoch

    # Training
    ## This line tells the model we're in 'learning mode'
    model.train()
    epoch_training_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs} Batch Size: {batch_size}, Transformer: {model_name}")
    for batch in train_iterator:
        optimizer.zero_grad()
        inputs = batch['input_ids'].squeeze(1).to(device)
        targets = inputs.clone()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_iterator.set_postfix({'Training Loss': loss.item()})
        epoch_training_loss += loss.item()
    avg_epoch_training_loss = epoch_training_loss / len(train_iterator)

    # Validation
    ## This line below tells the model to 'stop learning'
    model.eval()
    epoch_validation_loss = 0
    total_loss = 0
    valid_iterator = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
    with torch.no_grad():
        for batch in valid_iterator:
            inputs = batch['input_ids'].squeeze(1).to(device)
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss
            valid_iterator.set_postfix({'Validation Loss': loss.item()})
            epoch_validation_loss += loss.item()

    avg_epoch_validation_loss = epoch_validation_loss / len(valid_loader)

    end_time = time.time()  # End the timer for the epoch
    epoch_duration_sec = end_time - start_time  # Calculate the duration in seconds

    new_row = {'transformer': model_name,
               'batch_size': batch_size,
               'gpu': gpu,
               'epoch': epoch+1,
               'training_loss': avg_epoch_training_loss,
               'validation_loss': avg_epoch_validation_loss,
               'epoch_duration_sec': epoch_duration_sec}  # Add epoch_duration to the dataframe

    results.loc[len(results)] = new_row
    print(f"Epoch: {epoch+1}, Validation Loss: {total_loss/len(valid_loader)}")
    
```

## Result before finetuning:

**Input:** Why can camels survive for long without water?

**Output:** The answer is yes. If the sun's rays are shining, it will not be too hot to see them in their own eyes and therefore won't cause any damage or injury from these things that might happen during a day of sunshine on your face (e-mail me at: mamfjoe@gmail)

``` python
model config: 

GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-5): 6 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

```
### Result after finetuning: 
####  Number of epochs: 1  
Model size is similar: we didn't compress the model but we just finetuned:

`model name : 'distillgpt2-ft'`

---

**Input:** Why can camels survive for long without water?

**Output:** The camels are very thin and have a lot of moisture. They need to be kept in a dry environment. Some people like camels because they do not have enough water. They also need to keep their food and water.


**Actual ouput:** Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.

---

**Input:** When was Tomoaki Komorida born?

**Output:** Tomoaki Komorida was born on July 3, 1941.

**Actual Output:** Tomoaki Komorida was born on July 10,1981.

# Here We can see that model is using its previous weights and its hallucinating 

`4 epoch and temp = 0.5`

Why can camels survive for long without water? | Caramelization is the process by which a tiny plaque of oil comes from a tiny plaque that has bubbles and then rubs it out. The bubbles are then spread over time, and they become sticky and sticky. When they get wet you will also peel them and put them in your dish.

`4 epochs and temp = 0.1`

Why can camels survive for long without water? 

Caramel is the primary color of paint on the windshields when it comes to painting.  It's also a popular color for clothing, such as cotton lawns and velvet jackets.



| epoch | transformer | batch_size | gpu | training_loss | validation_loss | epoch_duration_sec |
|-------|-------------|------------|-----|---------------|-----------------|-------------------|
| 1     | distilgpt2  | 8          | 0   | 2.022359      | 1.895165        | 317.439226        |
| 1     | distilgpt2  | 8          | 0   | 1.661675      | 1.911140        | 317.462572        |
| 2     | distilgpt2  | 8          | 0   | 1.395908      | 2.002201        | 317.326218        |
| 3     | distilgpt2  | 8          | 0   | 1.162516      | 2.130988        | 317.301620        |
| 4     | distilgpt2  | 8          | 0   | 0.965772      | 2.325160        | 317.843375        |


  ## Why is this happening?

  Overgeneralization: LLMs are trained on vast datasets, but they do not "understand" facts the way humans do. They predict the next word based on statistical patterns rather than knowledge verification.

  Lack of Fact-Checking: LLMs lack internal mechanisms to verify the accuracy of their generated outputs. They rely on patterns in the data they were trained on, and if the data contains inconsistencies, outdated, or fictional information, it can lead to hallucination.

  Ambiguity in Input: Sometimes, LLMs generate incorrect or non-specific responses when the question is open-ended or not clear, though this isn't the case in your example.



### After 10epochs worked ryt on traing set 
**Input:** Why can camels survive for long without water? 

**Output:** Camels use the fat in their humps to keep them filled with energy and hydration so they cannot be outdoors longer than in a boat. 

# Overfiiting for unseen data 
**Question:** How can i learn swiming?

**Answer:** Swimming requires a constant change in posture. For instance, replace the daily stand up, then follow that for 30 minutes. This will allow you to practice all 3 steps forward.

Choose the right position on your own. For example, if you're riding a bike with 80 minutes of exercise, and be able to use 15 minutes.

Build the next Saturday morning (after a long walk) by 20 minutes. Avoid injury to cause damage to your joints. Keep gradually gain weight.

## Fine-tuning using Low order Rank Adaptation Method


## References:
- [PETL Research Paper](https://arxiv.org/pdf/1902.00751)
- [QLoRA Research Paper](https://arxiv.org/pdf/2305.14314)
- [LoRA Research Paper](https://arxiv.org/pdf/2106.09685)


# [Go to Approach ->](index.md)

## [<- Go to Fintuning Intro](finetuning.md)