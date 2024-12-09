# NLP_Transformers_finetunning_on_CommonsenseQA

We fine-tuned two pre-trained transformers on the CommonsenseQA dataset, which are BERT-base-uncased and RoBERTa-large. The code can be found in the directory of `BERT-base-uncased` and `RoBERTa-large` respectively. For each of them, we adopted two ways to fine-tune, including calling the Trainer API provided by Huggingface and writing training and evaluation loops manually. Apart from fine-tunning the pre-trained transformers, we also implemented and trained two traditional machine learning algorithms on the CommonsenseQA dataset for comparison, which are the Logistic Regression classifier and Neural Network classifier. The code of training these two algorithm can be found in the directory of `LogisticRegression-NeuralNetwork`.
