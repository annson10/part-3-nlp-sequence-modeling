# Part 3: NLP and Sequence Modeling Mini Project

## Objective
The objective of this project is to build an NLP pipeline to classify customer support messages by sentiment and compare traditional TF-IDF vectorization with an LSTM-based sequence model.

## Dataset
This project uses the `customer_support_text_classification.csv` dataset from:

https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

### Target Variable
- `sentiment_label`
  - positive
  - neutral
  - negative

### Input Column
- `customer_message`

## Methodology

### 1. Dataset Understanding
- Loaded the dataset
- Checked number of records
- Displayed sample text records
- Calculated average text length
- Visualized class distribution

### 2. Text Preprocessing
- Converted text to lowercase
- Removed special characters
- Tokenized text
- Applied padding for sequence models

### 3. Text Vectorization
- Used TF-IDF for baseline model
- Used tokenizer-based sequences for LSTM

### 4. Baseline Model
- Logistic Regression with TF-IDF

### 5. Sequence Model
- Embedding layer
- LSTM layer
- Dense layers
- Softmax output layer

### 6. Reflection
- RNN limitations
- LSTM memory mechanism
- Attention mechanism
- Importance of transformers

## Results
Generated files:
- `results/model_evaluation.csv`
- `results/sample_predictions.txt`

## Why Text Must Be Converted to Vectors
Machine learning models can only process numerical data. Text must therefore be converted into vectors that represent the importance or position of words. TF-IDF represents each document as a weighted numerical vector.      

## Attention and Transformer Reflection
### **Why RNNs struggle with long-term dependencies**
RNNs process sequences one step at a time. As sequences become longer, important information from earlier words may be forgotten due to vanishing gradients.

### **How LSTMs help with memory**
LSTMs use gates (input, forget, and output gates) to decide what information to keep or discard, allowing them to capture longer-term dependencies. 

- **Input gate** - decides what new information to store.
- **Forget gate** - decides what information to remove.
- **Output gate** - decides what information to pass to the next step.

These gates allow LSTMs to retain important information for longer periods and ignore irrelevant details.

### **What attention solves in sequence-to-sequence tasks**
In sequence-to-sequence tasks such as translation, relying on a single fixed-size context vector can cause information loss, especially for long sentences.

Attention solves this by allowing the model to look at all input words and focus on the most relevant ones when generating each output word. This improves performance and makes the model better at handling long sequences.

### **Why transformers are important in modern NLP and Generative AI**

Transformers are built entirely around self-attention mechanisms and do not rely on recurrence. This enables them to process all words in parallel, making training much faster and more scalable.

Transformers are highly effective at capturing long-range dependencies and understanding context.       

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt