from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from datasets import load_dataset, Dataset
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from  sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers.evaluation import NanoBEIREvaluator

tokenizer = Tokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
static_embedding = StaticEmbedding(tokenizer, embedding_dim=1024)

model = SentenceTransformer(modules=[static_embedding])

gooaq_dataset = load_dataset("sentence-transformers/gooaq", split="train")
gooaq_dataset_dict = gooaq_dataset.train_test_split(test_size=10_000, seed=12)
gooaq_train_dataset: Dataset = gooaq_dataset_dict["train"]
gooaq_eval_dataset: Dataset = gooaq_dataset_dict["test"]

tokenizer = Tokenizer.from_pretrained("google-bert/bert-base-uncased")
static_embedding = StaticEmbedding(tokenizer, embedding_dim=1024)
model = SentenceTransformer(modules=[static_embedding])

# Initialize the MNRL loss given the model
loss = MultipleNegativesRankingLoss(model)

run_name = "static-retrieval-mrl-en-v1"
# or 
# run_name = "static-similarity-mrl-multilingual-v1"

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=2048,
    per_device_eval_batch_size=2048,
    learning_rate=2e-1,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=1000,
    logging_first_step=True,
    run_name=run_name,  # Used if `wandb`, `tensorboard`, or `neptune`, etc. is installed
)

# Load an example pre-trained model to finetune further
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize the NanoBEIR Evaluator
evaluator = NanoBEIREvaluator()

# Run it on any Sentence Transformer model
evaluator(model)