# NLP Transformers finetuning on CommonsenseQA (BERT, RoBERTa, Classic ML Models)

Here is first part of our NLP final project, where we fine-tuned two pre-trained lightweight transformers on the CommonsenseQA dataset, which are BERT-base-uncased and RoBERTa-large. The code can be found in the directory of `BERT-base-uncased` and `RoBERTa-large` respectively. For each of them, we adopted two ways to fine-tune, including calling the Trainer API provided by Huggingface and writing training and evaluation loops manually. Apart from fine-tunning the pre-trained transformers, we also implemented and trained two traditional machine learning algorithms on the CommonsenseQA dataset for comparison, which are the Logistic Regression classifier and Neural Network classifier. The code for training these two algorithms can be found in the directory of `LogisticRegression-NeuralNetwork`.

Here are some introductions about what does each dir and file in this repo do:


## 
