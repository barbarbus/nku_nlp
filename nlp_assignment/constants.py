"""Assignment-mandated constants kept in one auditable location."""

DEFAULT_SEED = 42
WORD_VECTOR_SIZE = 100
BERT_MODEL_NAME = "google-bert/bert-base-uncased"
BERT_MAX_LENGTH = 64
BERT_EPOCHS = 3

# Stable display order matching the assignment tasks.
METHOD_ORDER = (
    "binary_bow",
    "word_frequency",
    "pretrained_glove",
    "word2vec_ag_news",
    "word2vec_nyt_train",
    "bert_base_uncased",
)

