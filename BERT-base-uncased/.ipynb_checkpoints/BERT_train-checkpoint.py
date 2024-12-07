import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    get_scheduler,
)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from accelerate import Accelerator
from tqdm.auto import tqdm
import evaluate

# Model checkpoint and batch size
model_checkpoint = "bert-base-uncased"
batch_size = 16


# Load the dataset
def load_and_preprocess_dataset():
    dataset = load_dataset("tau/commonsense_qa")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def preprocess_function(examples):
        # Extract the question stem
        first_sentences = examples["question"]

        # Extract all the answer texts (choices)
        second_sentences = [choice_dict["text"] for choice_dict in examples["choices"]]

        # Flatten lists for tokenization
        first_sentences = [stem for stem in first_sentences for _ in range(5)]
        second_sentences = [choice for choices in second_sentences for choice in choices]

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)

        # Group tokenized inputs by example (5 choices per question)
        grouped_inputs = {
            k: [v[i:i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()
        }
        return grouped_inputs

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("answerKey", "labels")
    encoded_dataset.set_format("torch")
    encoded_dataset = encoded_dataset.remove_columns(["id", "question", "question_concept", "choices"])

    return encoded_dataset, tokenizer


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        
        labels = [feature.pop("labels") for feature in features]

        # Map answerKey (e.g., "A", "B", ...) to numerical indices
        labels = torch.tensor(
            [["A", "B", "C", "D", "E"].index(label) for label in labels],
            dtype=torch.int64
        )

        # Determine batch size and number of choices
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flatten features for tokenization
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])  # Flatten the list of lists

        # Apply padding to the flattened features
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten to restore batch structure (batch_size, num_choices, sequence_length)
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        # Add back the labels as a tensor
        # batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["labels"] = labels

        return batch


# Training function
def training_function():
    # Load the dataset and tokenizer
    encoded_dataset, tokenizer = load_and_preprocess_dataset()

    # Initialize data collator
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # Prepare dataloaders
    train_dataloader = DataLoader(
        encoded_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        encoded_dataset["validation"], batch_size=batch_size, collate_fn=data_collator
    )

    # Load the model
    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Initialize accelerator
    accelerator = Accelerator()

    # Prepare model, dataloaders, and optimizer with the accelerator
    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    # Learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Initialize progress bar
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)

    # Load evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Evaluation loop
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in eval_dl:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.append(accelerator.gather(predictions))
        all_labels.append(accelerator.gather(batch["labels"]))

    # Concatenate predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=all_predictions, references=all_labels)
    f1 = f1_metric.compute(predictions=all_predictions, references=all_labels, average="weighted")
    precision = precision_metric.compute(predictions=all_predictions, references=all_labels, average="weighted")
    recall = recall_metric.compute(predictions=all_predictions, references=all_labels, average="weighted")

    eval_metrics = {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }

    accelerator.print("Evaluation results:", eval_metrics)


if __name__ == "__main__":
    training_function()
