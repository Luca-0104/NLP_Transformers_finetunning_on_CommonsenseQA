# NLP Transformers finetuning on CommonsenseQA (BERT, RoBERTa, Classic ML Models)

Here is first part of our NLP final project, where we fine-tuned two pre-trained lightweight transformers on the CommonsenseQA dataset, which are BERT-base-uncased and RoBERTa-large. The code can be found in the directory of `BERT-base-uncased` and `RoBERTa-large` respectively. For each of them, we adopted two ways to fine-tune, including calling the Trainer API provided by Huggingface and writing training and evaluation loops manually. Apart from fine-tunning the pre-trained transformers, we also implemented and trained two traditional machine learning algorithms on the CommonsenseQA dataset for comparison, which are the Logistic Regression classifier and Neural Network classifier. The code for training these two algorithms can be found in the directory of `LogisticRegression-NeuralNetwork`.


## Directory and File Descriptions
Here are some introductions about what does each dir and file in this repo do:

### `BERT-base-uncased/`
- **`BERT_finetune_CSQA_Trainer.ipynb`**: Contains our implementation of fine-tuning BERT on the CommonsenseQA dataset using HugginFace Trainer API.
- **`BERT_finetune_CSQA_notebook_launcher.ipynb`**: Contains our implementation of fine-tuning BERT on the CommonsenseQA dataset using our self-defined training loop and evaluation loop.

### `RoBERTA-large/`
- **`RoBERTa_finetune_CSQA_Trainer.ipynb`**: Contains our implementation of fine-tuning RoBERTa on the CommonsenseQA dataset using HugginFace Trainer API.
- **`RoBERTa_finetune_CSQA_notebook_launcher.ipynb`**: Contains our implementation of fine-tuning RoBERTa on the CommonsenseQA dataset using our self-defined training loop and evaluation loop.
- **`RoBERTa_finetune_CSQA_optimized.ipynb`**: This file contains our best accuracy reached on finetuning RoBERTa on the CommonsenseQA dataset using our self-defined training loop and evaluation loop. Notice that this file also contains our grid search hyperparameter tuning for RoBERTa. Moreover, this file also contains our optimization on RoBERTa finetuning with adding prefixes of 'Q' and 'A' to the questions and answer sentences.

### `LogisticsRegression-NeuralNetwork/`
- **`NLP_csqa_LogisticRegression_NeuralNetwork_model_training.ipynb`**: 
