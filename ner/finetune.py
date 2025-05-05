from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from seqeval.metrics import classification_report, f1_score
import pandas as pd
from typing import List, Union, Tuple

class FinetuneNer:
    def __init__(self):
        self.train_path = 'labelled.csv'
        self.model_checkpoint = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        # Add placeholders for whitespace
        self.whitespace_tokens = ["<SPACE>", "<TAB>"]
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.whitespace_tokens})

        # Label mapping (include 'O' for whitespace tokens)
        self.label_list = ['B-NAME', 'I-NAME', 'B-SSN', 'B-COMPANY', 'I-COMPANY', 'O']
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        # Load model and resize embeddings
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.label_list)
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.outputdir = "./ner-bert"

    def _mark_whitespace(self, text: Union[str, List[str], None]) -> List[str]:
        if text is None:
            return []
        marked = text.replace("\t", " <TAB> ")
        marked = marked.replace(" ", " <SPACE> ")
        return marked.split()

    def prepare_dataset(self) -> dict:
        df = pd.read_csv(self.train_path)
        print(df.head(0))
        ds = Dataset.from_pandas(df, preserve_index=False)
        return ds.train_test_split(test_size=0.1)

    def tokenize_and_align_labels(self, examples: dict) -> dict:
        texts = examples.get('text', [])
        labels = examples.get('label', [])
        token_lists = [self._mark_whitespace(t) for t in texts]

        tokenized = self.tokenizer(
            token_lists,
            is_split_into_words=True,
            truncation=True,
            padding=False
        )

        labels_batch = []
        for i, encoding in enumerate(tokenized.encodings):
            word_ids = encoding.word_ids
            prev_idx = None
            ex_labels = []
            # Safely handle label sequence
            tag_seq = labels[i] if i < len(labels) and isinstance(labels[i], list) else []
            for word_idx in word_ids:
                if word_idx is None:
                    ex_labels.append(-100)
                elif word_idx != prev_idx:
                    tag = tag_seq[word_idx] if word_idx < len(tag_seq) else 'O'
                    ex_labels.append(self.label2id.get(tag, self.label2id['O']))
                else:
                    ex_labels.append(-100)
                prev_idx = word_idx
            labels_batch.append(ex_labels)

        tokenized['labels'] = labels_batch
        return tokenized

    def compute_metrics(self, preds_and_labels: tuple) -> dict:
        predictions, labels = preds_and_labels
        preds = np.argmax(predictions, axis=2)
        true_preds, true_labels = [], []
        for pred_seq, label_seq in zip(preds, labels):
            p_seq, l_seq = [], []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    p_seq.append(self.label_list[p])
                    l_seq.append(self.label_list[l])
            true_preds.append(p_seq)
            true_labels.append(l_seq)
        return {
            'f1': f1_score(true_labels, true_preds),
            'report': classification_report(true_labels, true_preds)
        }

    def train(self):
        dataset = self.prepare_dataset()
        encoded = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=['text', 'label']
        )
        print(encoded['train'][0])
        training_args = TrainingArguments(
            output_dir=self.outputdir,
            # evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            save_total_limit=1
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded['train'],
            eval_dataset=encoded['test'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        trainer.evaluate()
        
    def infer(self, text: str) -> List[Tuple[str, str]]:
   
        tokens = self._mark_whitespace(text)
        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  return_tensors='pt')
        outputs = self.model(**encoding)
        preds = np.argmax(outputs.logits.detach().numpy(), axis=2)[0]
        word_ids = encoding.word_ids(batch_index=0)

        results = []
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                prev_word_idx = word_idx
                continue
            token = tokens[idx]
            label = self.id2label[preds[idx]]
            # Convert placeholder back to whitespace char
            if token == '<SPACE>': token = ' '
            if token == '<TAB>': token = '\t'
            results.append((token, label))
            prev_word_idx = word_idx
        return results

if __name__ == '__main__':
    ft = FinetuneNer()
    ft.train()
    sample = "Abigail Hooper 860-04-5019 Bond-Moore"
    print("Inference:", ft.infer(sample))
