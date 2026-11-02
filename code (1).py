"""Case Study 4: Sentiment Analysis on Product Reviews (AI201B CA2).

Install (Python 3.11/3.12):
    pip install torch "transformers>=4.40,<5" pandas numpy scikit-learn matplotlib
Run: python code.py       # downloads pretrained BERT, trains, writes output.pdf
Put dataset.csv beside this script. Internet is needed for the first model download.

Dataset: a balanced subset of the English Multilingual Amazon Reviews Corpus
(Keung et al., EMNLP 2020), via SetFit/amazon_reviews_multi_en on HuggingFace.
Mirror: https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en
Mirror license metadata: Apache-2.0. Original paper: https://arxiv.org/abs/2010.02573
CSV: 1,500 train + 300 validation + 300 test reviews per sentiment, seed 42.
Original source splits are preserved; duplicate normalized text is excluded.
Stars 1/2 -> negative, 3 -> neutral, 4/5 -> positive. Stars are NEVER model inputs.

Model: https://huggingface.co/google/bert_uncased_L-4_H-256_A-4 (Apache-2.0).
This is genuine pretrained BERT-Mini (4 layers, 256 hidden units), not BERT-Base.
All BERT weights are fine-tuned. LSTM weights start randomly. Both models use
the same fixed pretrained WordPiece tokenizer, text length and data splits.
No model weights are written, so the repository has only the requested 3 files.
"""

from pathlib import Path
import platform
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

FOLDER = Path(__file__).resolve().parent
MODEL = "google/bert_uncased_L-4_H-256_A-4"
REVISION = "387825ce42dbb39b87911cdf8e383ee3b25184f8"
NAMES = ["negative", "neutral", "positive"]
SEED = 42
MAX_LENGTH = 96
BATCH_SIZE = 32
BERT_EPOCHS = 4
LSTM_EPOCHS = 8


class ReviewLSTM(nn.Module):
    """A simple baseline: token embedding -> LSTM -> three class scores."""
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.classifier = nn.Linear(64, 3)

    def forward(self, input_ids, attention_mask):
        output, _ = self.lstm(self.embedding(input_ids))
        # Select the last REAL token's state, not the state after padding.
        last = attention_mask.sum(dim=1) - 1
        rows = torch.arange(len(input_ids), device=input_ids.device)
        return self.classifier(output[rows, last])


def make_loader(frame, tokenizer, shuffle=False):
    tokens = tokenizer(frame.text.tolist(), padding="max_length", truncation=True,
                       max_length=MAX_LENGTH, return_tensors="pt")
    data = TensorDataset(tokens["input_ids"], tokens["attention_mask"],
                         torch.tensor(frame.sentiment.map(dict(zip(NAMES, range(3)))).to_numpy()))
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)


def logits(model, ids, mask):
    output = model(input_ids=ids, attention_mask=mask)
    return output.logits if hasattr(output, "logits") else output


def synchronize(device):
    # GPUs run asynchronously; synchronize before recording elapsed time.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            scores = logits(model, ids.to(device), mask.to(device))
            actual.extend(labels.tolist())
            predicted.extend(scores.argmax(dim=1).cpu().tolist())
    return np.array(actual), np.array(predicted)


def train_model(model, train_loader, validation_loader, epochs, rate, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()
    history, best_score, best_weights = [], -1, None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for ids, mask, labels in train_loader:
            optimizer.zero_grad()
            scores = logits(model, ids.to(device), mask.to(device))
            loss = loss_function(scores, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        actual, predicted = predict(model, validation_loader, device)
        score = precision_recall_fscore_support(actual, predicted, average="macro",
                                               zero_division=0)[2]
        history.append((total_loss / len(train_loader.dataset), score))
        print(f"Epoch {epoch+1}/{epochs}: loss={history[-1][0]:.4f}; validation macro F1={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_weights = {name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()}
    synchronize(device)
    seconds = time.perf_counter() - start
    model.load_state_dict(best_weights)       # Select using validation, never test.
    return history, seconds


def evaluate(model, loader, device):
    # Warm up kernels; benchmark a full batched pass, without tokenization.
    first_ids, first_mask, _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        logits(model, first_ids.to(device), first_mask.to(device))
    synchronize(device)
    start = time.perf_counter()
    actual, predicted = predict(model, loader, device)
    synchronize(device)
    seconds = time.perf_counter() - start
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="macro", zero_division=0)
    return {"actual": actual, "predicted": predicted,
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision, "recall": recall, "f1": f1,
            "ms": seconds * 1000 / len(actual),
            "parameters": sum(p.numel() for p in model.parameters())}


def predict_review(model, tokenizer, text, device):
    """Example of using a trained model on a new review."""
    tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        scores = logits(model, tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
    return NAMES[scores.argmax(dim=1).item()]


def text_page(pdf, title, paragraphs, monospace=False):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=17, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        content = str(paragraph) if monospace else textwrap.fill(str(paragraph), width=88)
        fig.text(0.08, y, content, fontsize=9 if monospace else 10, va="top",
                 family="monospace" if monospace else "sans-serif", linespacing=1.45)
        y -= 0.025 * (content.count("\n") + 1) + 0.022
    fig.text(0.08, 0.035, "AI201B | CA2 | Case Study 4", fontsize=9, color="gray")
    pdf.savefig(fig)
    plt.close(fig)


def create_report(data, test, results, histories, times, device):
    with PdfPages(FOLDER / "output.pdf") as pdf:
        text_page(pdf, "Case Study 4: Review Sentiment", [
            "Aim: classify Amazon product review text as negative, neutral or positive; "
            "fine-tune a HuggingFace pretrained BERT, compare an LSTM baseline, and "
            "discuss transfer learning efficiency and deployment constraints.",
            "Source: Keung, Lu, Szarvas and Smith (2020), The Multilingual Amazon "
            "Reviews Corpus, English subset. https://arxiv.org/abs/2010.02573. "
            "Downloaded via https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en "
            "(mirror metadata: Apache-2.0). No synthetic reviews are included.",
            f"Data: {len(data):,} reviews; " + "; ".join(
                f"{split}={len(group):,}" for split, group in data.groupby("split")) + ". "
            "Each split is balanced across three sentiments. Within negative/positive, "
            "both associated star ratings are sampled equally. Sampling seed: 42.",
            "Original train/validation/test membership is preserved. Duplicate normalized "
            "review text is removed across splits. Rating-derived labels: 1/2 stars "
            "negative; 3 neutral; 4/5 positive. Stars never enter either model. "
            "A three-star review may express mixed feelings rather than true neutrality.",
            f"BERT: {MODEL}. Genuine Google pretrained BERT-Mini, four layers and "
            f"256 hidden units; every parameter is fine-tuned for {BERT_EPOCHS} epochs, "
            "AdamW learning rate 5e-5. This experiment does not benchmark BERT-Base.",
            f"LSTM: random 64-dimensional embeddings -> one 64-unit LSTM -> linear "
            f"3-class head; {LSTM_EPOCHS} epochs, AdamW learning rate 1e-3. "
            "Same fixed WordPiece tokenizer as BERT, but no pretrained embedding weights.",
            f"Both: maximum {MAX_LENGTH} tokens, batch size {BATCH_SIZE}, cross-entropy "
            "loss, weight decay 0.01, gradient clipping 1.0. Best epoch is selected "
            "using validation macro F1; the held-out test set is evaluated afterward.",
        ])

        rows = []
        for name, result in results.items():
            rows.append([name, f"{result['accuracy']:.3f}", f"{result['precision']:.3f}",
                         f"{result['recall']:.3f}", f"{result['f1']:.3f}",
                         f"{times[name]:.1f}", f"{result['ms']:.2f}"])
        fig, axis = plt.subplots(figsize=(11.7, 5))
        axis.axis("off")
        axis.set_title("Held-out test comparison", fontsize=17, pad=25)
        table = axis.table(cellText=rows, colLabels=["Model", "Accuracy", "Macro P",
                           "Macro R", "Macro F1", "Fit + val (s)", "ms/review"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        fig.text(0.08, 0.17, "Timing: one warmed-up batched test pass on the same device.\n"
                 "Includes batch transfers and prediction collection; excludes tokenization/model download.\n"
                 "ms/review is amortized throughput, not single-request latency. Balanced chance accuracy = 1/3.", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.7, 5))
        for name, history in histories.items():
            values = np.array(history)
            epochs = np.arange(1, len(history)+1)
            axes[0].plot(epochs, values[:, 0], "o-", label=name)
            axes[1].plot(epochs, values[:, 1], "o-", label=name)
        axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy", title="Learning curves")
        axes[1].set(xlabel="Epoch", ylabel="Validation macro F1", title="Checkpoint selection")
        for axis in axes:
            axis.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.5))
        for axis, (name, result) in zip(axes, results.items()):
            matrix = confusion_matrix(result["actual"], result["predicted"], labels=[0, 1, 2])
            ConfusionMatrixDisplay(matrix, display_labels=NAMES).plot(ax=axis, colorbar=False)
            axis.set_title(name)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for name, result in results.items():
            report = classification_report(result["actual"], result["predicted"],
                labels=[0, 1, 2], target_names=NAMES, digits=3, zero_division=0)
            text_page(pdf, f"{name}: per-class test results", [report], monospace=True)

        bert = results["BERT-Mini"]
        wrong = np.flatnonzero(bert["actual"] != bert["predicted"])[:4]
        examples = [f"True: {NAMES[bert['actual'][i]]}; predicted: "
                    f"{NAMES[bert['predicted'][i]]}. Review: {test.iloc[i].text[:260]}"
                    for i in wrong]
        text_page(pdf, "BERT: observed test errors", examples + [
            "These are the first errors in test order, not hand-picked successes. "
            "Rating labels, mixed sentiment, sarcasm and review truncation can all "
            "cause disagreement; an error example alone does not prove its cause."])

        lstm = results["LSTM"]
        text_page(pdf, "Transfer learning and efficiency", [
            f"Observed test macro F1: BERT-Mini={bert['f1']:.3f}; LSTM={lstm['f1']:.3f}; "
            f"difference={bert['f1']-lstm['f1']:+.3f}. "
            "This is a single seeded comparison on the included sample, not proof "
            "that either architecture always wins.",
            "BERT begins with language representations learned during pretraining; "
            "fine-tuning adapts them using fewer task-specific labels than learning "
            "language from random weights. The LSTM learns its embeddings and sequence "
            "weights from this training set. Both reuse the fixed pretrained tokenizer. "
            "A random-initialized BERT control and multiple sample sizes would be "
            "needed to isolate and quantify the benefit of pretraining itself.",
            f"BERT-Mini has {bert['parameters']:,} trainable parameters; LSTM has "
            f"{lstm['parameters']:,}. FP32 weight storage estimates: "
            f"{bert['parameters']*4/1024**2:.1f} MiB vs "
            f"{lstm['parameters']*4/1024**2:.1f} MiB. "
            "These are parameter-only estimates, not measured RAM. Gradients, AdamW "
            "optimizer states, activations and the runtime require additional memory.",
            f"Measured fit + validation: {times['BERT-Mini']:.1f}s for BERT-Mini "
            f"({BERT_EPOCHS} epochs); {times['LSTM']:.1f}s for LSTM ({LSTM_EPOCHS} epochs). "
            "Different epoch budgets and parameter counts make this a practical "
            "baseline comparison, not an equal-compute experiment. Offline pretraining "
            "cost and first-time downloads are excluded.",
            "BERT uses bidirectional self-attention; attention cost grows quadratically "
            "with sequence length. LSTM processes tokens sequentially and has fewer "
            "parameters here. Shortening the token limit saves work but may discard "
            "important sentiment near the end of a review.",
            "Model source and architecture: "
            "https://huggingface.co/google/bert_uncased_L-4_H-256_A-4. "
            "Turc et al. (2019), Well-Read Students Learn Better, "
            "https://arxiv.org/abs/1908.08962.",
        ])
        text_page(pdf, "Deployment constraints and reproduction", [
            "Latency and resources: measure end-to-end p50/p95 latency on the target "
            "CPU/GPU including tokenization, queueing and model load. Batch throughput "
            "reported here is not a serving SLA. Cache one loaded model and tokenizer; "
            "test quantization/distillation for smaller deployments before accepting "
            "any accuracy tradeoff.",
            "Input limits: this is an English-only sample with a 96-token limit. "
            "Long reviews, other languages, sarcasm and new product categories need "
            "separate evaluation. Original source splits do not guarantee product- "
            "or reviewer-disjoint evaluation because those identifiers are absent "
            "from the mirror. Balanced sampling does not reflect live class priors.",
            "Reliability: monitor per-class F1 and changing language/product patterns. "
            "Validate and calibrate probabilities on fresh data before using a "
            "confidence threshold. Send uncertain/high-impact cases for human review. "
            "Do not treat an uncalibrated softmax as trustworthy certainty.",
            "Privacy: minimize review logging, redact personal information, set retention "
            "limits and restrict access. Package exact model/tokenizer revisions and "
            "tested dependency versions; assess dataset/model terms for the intended use.",
            f"Execution: {platform.system()} {platform.release()}, {platform.machine()}, "
            f"device={device}, CPU threads={torch.get_num_threads()}. "
            f"Python {platform.python_version()}; torch {torch.__version__}; "
            f"transformers {transformers.__version__}; pandas {pd.__version__}; "
            f"scikit-learn {sklearn.__version__}; Matplotlib {matplotlib.__version__}. "
            "Seeds fix sampling and initialization; hardware kernels can still vary.",
            "Reproduce: install packages listed at the top of code.py, place dataset.csv "
            "beside it, then run python code.py. Training writes only output.pdf; "
            "the trained models stay in memory. predict_review() demonstrates inference. "
            "A production service would separately persist and version trained weights.",
            "HuggingFace fine-tuning reference: "
            "https://huggingface.co/docs/transformers/tasks/sequence_classification. "
            "PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html",
        ])


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    data = pd.read_csv(FOLDER / "dataset.csv")
    assert data.sentiment.isin(NAMES).all() and data.text.notna().all()
    assert data.text.str.strip().str.lower().nunique() == len(data), "Duplicate review text"
    splits = {name: data[data.split == name].reset_index(drop=True)
              for name in ["train", "validation", "test"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    loaders = {name: make_loader(frame, tokenizer, shuffle=(name == "train"))
               for name, frame in splits.items()}
    results, histories, times = {}, {}, {}
    for name in ["BERT-Mini", "LSTM"]:
        torch.manual_seed(SEED)
        if name == "BERT-Mini":
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, revision=REVISION, num_labels=3,
                id2label=dict(enumerate(NAMES)), label2id=dict(zip(NAMES, range(3))))
        else:
            model = ReviewLSTM(len(tokenizer))
        print(f"\nTraining {name}", flush=True)
        histories[name], times[name] = train_model(model, loaders["train"], loaders["validation"],
            BERT_EPOCHS if name == "BERT-Mini" else LSTM_EPOCHS,
            5e-5 if name == "BERT-Mini" else 1e-3, device)
        results[name] = evaluate(model, loaders["test"], device)
        print(f"{name}: accuracy={results[name]['accuracy']:.3f}, macro F1={results[name]['f1']:.3f}", flush=True)
        print("New-review demo:", predict_review(model, tokenizer,
              "The product works perfectly and I am happy with it.", device), flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    create_report(data, splits["test"], results, histories, times, device)
    print("Created output.pdf", flush=True)


if __name__ == "__main__":
    main()
