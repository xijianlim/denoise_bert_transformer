from datasets import load_dataset
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader,TensorDataset
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
import os
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import time
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt



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


class NoisyTransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_classes, nhead=8, dim_feedforward=8192, modes=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.modes = modes

        # Fourier denoising parameters
        self.scale_real = nn.Parameter(torch.randn(modes, embedding_dim) * 0.02)
        self.scale_imag = nn.Parameter(torch.randn(modes, embedding_dim) * 0.02)
        self.project = spectral_norm(nn.Linear(embedding_dim, embedding_dim))

        # Learnable residual weights (LayerScale)
        self.scale_attn = nn.Parameter(torch.ones(1))     # for attention
        self.scale_denoise = nn.Parameter(torch.ones(1))  # for denoiser

        # Transformer-style layers
        self.self_attn = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Final feedforward
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embedding_dim),
        )

        # Learnable residual gate
        self.gate = nn.Parameter(torch.tensor(0.5))

        # Learnable residual weights (LayerScale)
        self.scale_attn = nn.Parameter(torch.ones(1))     # for attention
        self.scale_denoise = nn.Parameter(torch.ones(1))  # for denoiser

    def fft_denoise(self, z):
        """
        Vectorized Fourier denoising.
        z: [B, D]
        """
        B, D = z.shape
        z_freq = torch.fft.fft(z, dim=1)  # [B, D], complex

        filters = torch.complex(self.scale_real, self.scale_imag)  # [modes, D]
        filters = filters.unsqueeze(0)  # [1, modes, D]

        z_selected = z_freq[:, :self.modes]  # [B, modes]
        z_filtered = torch.zeros_like(z_freq)

        for k in range(self.modes):
            z_filtered[:, k] = z_selected[:, k] * filters[0, k].mean()

        z_spatial = torch.fft.ifft(z_filtered, dim=1).real  # [B, D]
        return self.project(z_spatial)  # Linear projection

    def forward(self, x_feat, z_prev, W_embed, alpha_bar_t=None):
        """
        x_feat: [B, L, D] - encoded input features
        z_prev: [B, D]    - previous latent
        W_embed: [C, D]   - class embedding matrix
        """
        B = z_prev.size(0)

        # --- Step 1: Add noise
        z_noisy = z_prev

        z = z_noisy.unsqueeze(1)  # [B, 1, D]
        z_residual = z.clone()

       # Step 2: Fourier-domain denoising with LayerScale
        z_denoised = self.fft_denoise(z.squeeze(1)).unsqueeze(1)  # [B, 1, D]
        z = z + self.scale_denoise * z_denoised  # LayerScale on denoising

        # Step 3: Self-attention with LayerScale
        z1, _ = self.self_attn(z, z, z)
        z = self.norm1(z + self.scale_attn * z1)

        # Step 4: Cross-attention
        z2, _ = self.cross_attn(z, x_feat, x_feat)
        z = self.norm2(z + z2)

        # Step 5: Feedforward
        z3 = self.ff(z)
        z = self.norm3(z + z3)

        # Step 6: Gated residual connection
        gate = torch.sigmoid(self.gate)
        z = gate * z + (1 - gate) * z_residual

        z = z.squeeze(1)  # [B, D]

        # Step 7: Logits + reconstruction
        logits = z @ W_embed.T  # [B, C]
        p = F.softmax(logits, dim=1)
        z_next = p @ W_embed    # [B, D]


        return z_next, logits

class NoPropDTDecoder(nn.Module):
    def __init__(self, num_classes, embedding_dim, T, eta, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta
        self.attn_pool = AttentionPool(embedding_dim)

        # Learnable class embedding matrix
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Decoder blocks (each with internal noise injection)
        self.blocks = nn.ModuleList([
            NoisyTransformerDecoderBlock(embedding_dim, num_classes)
            for _ in range(T)
        ])

        # Fusion + classifier head
        self.fuse_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

        # Cosine noise schedule
        t = torch.arange(1, T + 1, dtype=torch.float32)
        alpha_t = torch.cos(t / T * (math.pi / 2)) ** 2
        alpha_bar = torch.cumprod(alpha_t, dim=0)
        snr = alpha_bar / (1 - alpha_bar)
        snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)
        snr_diff = snr - snr_prev

        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('snr_diff', snr_diff)

    def forward_denoise(self, x_feat, z_prev, t):
        """
        Denoise latent z_prev using decoder at time t.
        Noise is injected inside each decoder block.
        """
        if x_feat.dim() == 2:
            x_feat = x_feat.unsqueeze(1)
        return self.blocks[t](x_feat, z_prev, self.W_embed, alpha_bar_t=self.alpha_bar[t])[0]

    def classify(self, x_feat, z_final):
        """
        Fusion of x_feat and final denoised z → prediction
        """
        if x_feat.dim() == 3:
            x_feat = self.attn_pool(x_feat)
        fused = torch.cat([x_feat, z_final], dim=-1)  # [B, 2D]
        h = self.fuse_mlp(fused)  # [B, 128]
        return self.classifier(h)

    def inference(self, x_feat):
        """
        Run full denoising from pure noise and classify.
        """
        B = x_feat.size(0)
        z = torch.randn(B, self.embedding_dim, device=x_feat.device)

        if x_feat.dim() == 2:
            x_feat = x_feat.unsqueeze(1)

        for t in range(self.T):
            z = self.forward_denoise(x_feat, z, t)

        return self.classify(x_feat, z)


def train_nopropdt(model, train_loader, test_loader, epochs, lr, weight_decay, best_metric='f1'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'val_acc': []}
    best_score = -float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            uy = model.W_embed[y]  # clean class embeddings
            z = torch.randn_like(uy)  # initial noisy latent

            total_loss = 0

            for t in range(model.T):
                z_pred = model.forward_denoise(x, z, t)  # noise injected inside the block
                loss_l2 = F.mse_loss(z_pred, uy)
                loss = 0.5 * model.eta * model.snr_diff[t] * loss_l2

                # Optionally apply classification at every step (deep supervision)
                if t == model.T - 1:
                    logits = model.classify(x, z_pred)
                    loss_ce = F.cross_entropy(logits, y)
                    loss_kl = 0.5 * uy.pow(2).sum(dim=1).mean()
                    loss = loss + loss_ce + 0.001 * loss_kl  # optional KL regularization

                total_loss += loss
                z = z_pred.detach()  # carry forward the denoised latent

            optimizer.zero_grad()
            total_loss.backward()
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

        # Metrics
        epoch_time = time.time() - start_time
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"TrainAcc: {100 * train_acc:.2f}% | ValAcc: {100 * val_acc:.2f}%")
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

    print(f"\nFinal Test Accuracy: {100 * val_acc:.2f}%")   

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


    no_prob_model = NoPropDTDecoder(num_classes=14, embedding_dim=embedding_dim, T=8, eta = eta, dropout=dropout).to(device)
    train_nopropdt(no_prob_model, train_loader, test_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, best_metric='f1')

    print_test_preds(no_prob_model, test_loader)


    # save_path = "nopropdt_decoder_pubmed.pth"  # or /dbfs/tmp/...
    # torch.save(no_prob_model.state_dict(), save_path)
    # print(f"Saved model to: {save_path}")



if __name__ == "__main__":
    run()