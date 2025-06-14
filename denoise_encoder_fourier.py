from datasets import load_dataset
import torch
import pandas as pd
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader,TensorDataset
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
import os
import torch.nn.functional as F
import time
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


batch_size = 128
num_classes = 14  # DBpedia has 14 classes
lr = 1e-3
eta=0.1
dropout=0.1
epochs = 10
weight_decay = 1e-3
model_path='models/bert-based-uncased/'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = model.config.hidden_size

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def encode_and_embed(example):
    tokens = tokenizer(
        example["abstract_text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)
        cls_embedding = output.last_hidden_state[:, 0, :]  # [B, D]

    return {
        "embedding": cls_embedding.squeeze(0).cpu().numpy(),
        "label": example["label"]  # now a plain int
    }
# Load the DBpedia dataset (14 classes)
# 1. Load the DBpedia dataset

# Embed function
def load_dataset(cache_dir="data/db14/cache_embeds", force_recompute=True,
  test_size=0.2, random_seed=42, dataset_path="data/db14/", sample_size=10000):
    # Check if cached dataset exists
    if os.path.exists(cache_dir) and not force_recompute:
        print("🔄 Loading cached embedded dataset...")
        encoded_dataset = load_from_disk(cache_dir)
    else:
        print("⚙️  Computing embeddings from scratch...")
        df = pd.read_csv(f"{dataset_path}train.csv")


        df_sampled = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        hf_dataset = Dataset.from_pandas(df_sampled)

        # Apply tokenizer/embedding
        encoded_dataset = hf_dataset.map(
            encode_and_embed,
            remove_columns=hf_dataset.column_names,
            load_from_cache_file=False
        )

        # ✅ Save mapped embeddings to disk
        print(f"💾 Saving mapped dataset to: {cache_dir}")
        encoded_dataset.save_to_disk(cache_dir)

    # Split into train/test
    split = encoded_dataset.train_test_split(test_size=test_size, seed=random_seed)
    train_dataset = split["train"]
    test_dataset = split["test"]

    train_tensor_dataset = dataset_to_tensors(train_dataset)
    test_tensor_dataset = dataset_to_tensors(test_dataset)

    return train_tensor_dataset, test_tensor_dataset


def dataset_to_tensors(dataset):
    embeddings = torch.tensor([example["embedding"] for example in dataset], dtype=torch.float32)
    labels = torch.tensor([example["label"] for example in dataset], dtype=torch.long)
    return TensorDataset(embeddings, labels)



class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, L, D]
        scores = self.attn(x)  # [B, L, 1]
        weights = F.softmax(scores, dim=1)  # [B, L, 1]
        pooled = (weights * x).sum(dim=1)  # [B, D]
        return pooled


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled

class NoisyTransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, nhead=8, dim_feedforward=1024, dropout=0.2, modes=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=nhead, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Fourier parameters
        self.modes = modes
        self.scale_real = nn.Parameter(torch.randn(modes, embedding_dim) * 0.02)
        self.scale_imag = nn.Parameter(torch.randn(modes, embedding_dim) * 0.02)
        self.project = spectral_norm(nn.Linear(embedding_dim, embedding_dim)) # Final projection after IFFT

        self.bn_z = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.5))

        # Learnable residual weights (LayerScale)
        self.scale_attn = nn.Parameter(torch.ones(1))     # for attention
        self.scale_denoise = nn.Parameter(torch.ones(1))  # for denoiser

    def fft_denoise(self, z):
        """
        z: [B, D] — latent input
        Returns: [B, D] — denoised vector via Fourier filtering
        """
        B, D = z.shape
        z_freq = torch.fft.fft(z, dim=1)  # [B, D], complex

        filters = torch.complex(self.scale_real, self.scale_imag)  # [modes, D]
        z_filtered = torch.zeros_like(z_freq)

        for k in range(self.modes):
            z_filtered[:, k] = z_freq[:, k] * filters[k].mean()

        z_spatial = torch.fft.ifft(z_filtered, dim=1).real  # [B, D]
        return self.project(z_spatial)  # [B, D]

    def forward(self, x_feat, z_prev, alpha_bar_t=None):
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)

        B, D = z_prev.shape

        if alpha_bar_t is not None:
            if alpha_bar_t.dim() == 0:
                alpha_bar_t = alpha_bar_t.expand(B).unsqueeze(1)
            elif alpha_bar_t.dim() == 1:
                alpha_bar_t = alpha_bar_t.unsqueeze(1)

        
        z_noisy = z_prev

        z = z_noisy.unsqueeze(1)
        z_residual = z.clone()

        z_attn, _ = self.self_attn(z, x_feat, x_feat)
        z = self.norm1(z + self.scale_attn * self.dropout(z_attn))

        # Use Fourier denoising here
        z_denoised = self.fft_denoise(z.squeeze(1)).unsqueeze(1)
        z = self.norm2(z + self.scale_denoise * self.dropout(z_denoised))

        gate = torch.sigmoid(self.gate)
        z = gate * z + (1 - gate) * z_residual
        z = self.bn_z(z.squeeze(1)).unsqueeze(1)

        return z.squeeze(1)



class NoPropDTEncoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, nhead=8, T=4, eta=0.03, dropout=0.2, dim_feedforward=1024, use_averaging=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta
        self.input_dropout = nn.Dropout(dropout)
        self.use_averaging = use_averaging

        self.attn_pool = AttentionPool(embedding_dim)

        self.blocks = nn.ModuleList([
            NoisyTransformerEncoderBlock(embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(T)
        ])

        self.fuse_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

        t = torch.arange(1, T + 1, dtype=torch.float32)
        alpha_t = torch.cos(t / T * (math.pi / 2)) ** 2
        alpha_bar = torch.cumprod(alpha_t, dim=0)
        snr = alpha_bar / (1 - alpha_bar)
        snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)
        snr_diff = snr - snr_prev

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("snr_diff", snr_diff)

    def forward_denoise(self, x_feat, z_prev, t):
        #alpha_bar_t = self.alpha_bar[t] if self.training else None
        return self.blocks[t](x_feat, z_prev, self.alpha_bar[t])

    def classify(self, z):
        h = self.fuse_mlp(z)
        return self.classifier(h)

    def inference(self, x_feat):
        x_feat = self.input_dropout(x_feat)
        z = self.attn_pool(x_feat)

        logits_sum = 0
        for t in range(self.T):
            z = self.forward_denoise(x_feat, z, t)
            logits_sum += self.classify(z)

        return logits_sum / self.T if self.use_averaging else self.classify(z)


def train_noprop_encoder(model, train_loader, test_loader, epochs, lr, weight_decay, best_metric='f1', save_path="best_model.pt"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'val_acc': []}
    best_score = -float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()

        for t in range(model.T):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                B = x.size(0)

                with torch.no_grad():
                    uy = model.attn_pool(x)

                alpha_bar_t = model.alpha_bar[t]
                if alpha_bar_t.dim() == 0:
                    alpha_bar_t = alpha_bar_t.expand(B).unsqueeze(1)

                noise = torch.randn_like(uy)
                z_t = torch.sqrt(alpha_bar_t) * uy + torch.sqrt(1 - alpha_bar_t) * noise
                z_pred = model.blocks[t](x, z_t, alpha_bar_t=alpha_bar_t)
                loss_l2 = F.mse_loss(z_pred, uy)
                loss = 0.5 * model.eta * model.snr_diff[t] * loss_l2

                if t == model.T - 1:
                    logits = model.classify(z_pred)
                    loss_ce = F.cross_entropy(logits, y, label_smoothing=0.1)
                    loss_kl = 0.5 * uy.pow(2).sum(dim=1).mean()
                    loss += loss_ce + 0.001 * loss_kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # --- Evaluation ---
        model.eval()

        def evaluate(loader):
            correct, total = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                    preds = model.inference(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            acc = correct / total
            return acc, all_preds, all_labels

        train_acc, _, _ = evaluate(train_loader)
        val_acc, all_preds, all_labels = evaluate(test_loader)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - start_time
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Epoch {epoch}/{epochs} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")
        print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # --- Save best model by selected metric ---
        metric_map = {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': val_acc}
        current_score = metric_map[best_metric.lower()]
        if current_score > best_score:
            best_score = current_score
            best_state_dict = model.state_dict()

    # --- Reload best model ---
    print(f"\n✅ Restoring best model based on {best_metric.upper()} = {best_score:.4f}")
    model.load_state_dict(best_state_dict)


    # --- Plot ---
    plt.figure()
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_test_preds(model, test_loader):
    model.eval()    
    start_time = time.time()
    # Get one batch
    x_batch, y_true = next(iter(test_loader))
    x_batch = x_batch.to(device)
    y_true = y_true.to(device)

    # ✅ Ensure input is 3D: [B, L, D]
    if x_batch.dim() == 2:
        x_batch = x_batch.unsqueeze(1)

    with torch.no_grad():
        logits = model.inference(x_batch)  # returns [B, C]
        y_pred = logits.argmax(dim=1)
    end_time = time.time()
    print(f"Time taken for inference on one batch: {end_time - start_time:.4f} seconds")
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    for i in range(len(y_true)):
        print(f"Sample {i}: Predicted = {y_pred[i]} | Actual = {y_true[i]}")


def run():
    
    train_tensor_dataset, test_tensor_dataset = load_dataset(force_recompute=False)
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_tensor_dataset,  batch_size=batch_size)


    no_prob_model = NoPropDTEncoder(
    num_classes=14,
    embedding_dim=embedding_dim,
    nhead=3,
    dim_feedforward=2048,
    use_averaging=False,
    dropout=dropout,
    eta=eta).to(device)

    train_noprop_encoder(no_prob_model, train_loader, test_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)

    print_test_preds(no_prob_model, test_loader)




if __name__ == "__main__":
    run()