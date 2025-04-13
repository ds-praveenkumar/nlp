from datasets import load_dataset, ClassLabel, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from seqeval.metrics import classification_report, f1_score
import pandas as pd

class FinetuneNer:
    def __init__( self ):
        self.train_path = 'labelled.csv'
        self.model_checkpoint = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.label_list = ['B-NAME', 'I-NAME', 'B-SSN', 'B-COMPANY' ,'I-COMPANY']
        self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_checkpoint, num_labels=len(self.label_list)
                )
        self.outputdir = "./ner-bert"

    def prepare_dataset( self ):
        
        df = pd.read_csv( self.train_path )
        ds = Dataset.from_pandas(df)
        ds['train'], ds['validation'] = ds.train_test_split(.1).values()
        return ds
    
    def tokenize_and_align_labels(self, example):
        tokenized_inputs = self.tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(self.label_list.index(example["ner_tags"][word_idx]))
            else:
                labels.append(self.label_list.index(example["ner_tags"][word_idx]) if True else -100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics( self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return {
            "f1": f1_score(true_labels, true_predictions),
            "report": classification_report(true_labels, true_predictions)
        }
    
    def train(self):
        # Define training arguments
        dataset = self.prepare_dataset()
        encoded_dataset = dataset.map(self.tokenize_and_align_labels, batched=True)
        training_args = TrainingArguments(
            output_dir=self.outputdir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            save_total_limit=2,
        )

        # Data collator for dynamic padding
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train
        trainer.train()

        # Evaluate
        trainer.evaluate()

if __name__ == '__main__':
    ft = FinetuneNer()
    ft.train()