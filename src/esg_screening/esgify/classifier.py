"""ESGify model wrapper (https://github.com/sb-ai-lab/ESGify)."""

from collections import OrderedDict

import torch
from transformers import AutoTokenizer, MPNetModel, MPNetPreTrainedModel

MODEL_NAME = "ai-lab/ESGify"


def mean_pooling(token_embeddings, attention_mask):
    """Average the token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class ESGify(MPNetPreTrainedModel):
    """MPNet encoder with a custom head that scores 47 ESG risk classes."""

    def __init__(self, config):
        super().__init__(config)
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        self.id2label = config.id2label
        self.label2id = config.label2id
        self.classifier = torch.nn.Sequential(OrderedDict([
            ("norm", torch.nn.BatchNorm1d(768)),
            ("linear", torch.nn.Linear(768, 512)),
            ("act", torch.nn.ReLU()),
            ("batch_n", torch.nn.BatchNorm1d(512)),
            ("drop_class", torch.nn.Dropout(0.2)),
            ("class_l", torch.nn.Linear(512, 47)),
        ]))
        # Required by transformers v5 for custom models (the original ESGify README targets v4)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        outputs = self.mpnet(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(mean_pooling(outputs["last_hidden_state"], attention_mask))
        # Sigmoid: each class gets an independent 0-1 score (multi-label)
        return torch.sigmoid(logits)


def load_model(model_name=MODEL_NAME):
    """Download (first run only) and load the ESGify model and its tokenizer."""
    model = ESGify.from_pretrained(model_name)
    model.eval()  # inference mode: freezes BatchNorm statistics and disables Dropout
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def predict_scores(texts, model, tokenizer, batch_size=16):
    """Return a (len(texts), 47) tensor with the score of every class for every text.

    Texts are processed in batches so that long documents don't need all chunks in memory at once.
    Texts of similar length are batched together: every text in a batch is padded to the
    longest one, so mixing short and long texts wastes computation.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]
    batches = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            sorted_texts[start:start + batch_size],
            add_special_tokens=True,
            max_length=512,  # the model's limit; longer texts are truncated
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            batches.append(model(**inputs))
    sorted_scores = torch.cat(batches)
    scores = torch.empty_like(sorted_scores)
    scores[order] = sorted_scores  # back to the original order of `texts`
    return scores


def classify(texts, model, tokenizer, top_k=3):
    """Return the top_k ESG labels with scores for each text."""
    scores = predict_scores(texts, model, tokenizer)
    top = torch.topk(scores, k=top_k)
    return [
        [(model.id2label[i], round(s, 3)) for i, s in zip(idx, val)]
        for idx, val in zip(top.indices.tolist(), top.values.tolist())
    ]
