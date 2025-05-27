import torch
import torch.nn as nn
import math



class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (B, dim)

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t_embed):
        return self.embed(t_embed)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=6000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=1, dropout=0.0):  # Reduced to 1 head and no dropout
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):  # No dropout
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, seq_len, dim=64, depth=1, heads=1, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(4, dim)
        self.pos_enc = PositionalEncoding(dim, max_len=seq_len)
        self.timestep_embed = SinusoidalTimestepEmbedding(dim)
        self.t_embed_proj = TimestepEmbedder(dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, hidden_dim) for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, timesteps):
        x = self.embedding(x)  # (B, L, dim)
        x = self.pos_enc(x)
        t_emb = self.t_embed_proj(self.timestep_embed(timesteps))  # (B, dim)
        x = x + t_emb.unsqueeze(1)  # Add timestep

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)  # Global average pooling
        return x  # (B, num_classes)




import torch
import torch.nn as nn
import torch.nn.functional as F
from layer_HGNN import HGNN_conv

class DNA_HGNN(nn.Module):
    def __init__(self, in_ch=4, hidden_ch=16, num_classes=2, dropout=0.1):
        super(DNA_HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, hidden_ch)
        self.hgc2 = HGNN_conv(hidden_ch, num_classes)

    def forward(self, x, G):
        """
        x: Tensor of shape (N, F) - Feature input (e.g., one-hot or k-mer)
        G: Tensor of shape (N, N) - Normalized graph (manually set to 10x10)
        """
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)
        return x




class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.input_dim = input_dim

    def forward(self, hgnn_feat, trans_feat):
        combined = torch.stack([hgnn_feat, trans_feat], dim=1)  # (B, 2, D)
        attn_scores = torch.matmul(combined, combined.transpose(-2, -1)) / (combined.size(-1) ** 0.5)
        attn_weights = self.softmax(attn_scores)  # (B, 2, 2)
        fused = torch.matmul(attn_weights, combined).mean(dim=1)  # (B, D)
        return fused


class FusionGeneExpressionModel(nn.Module):
    def __init__(self, seq_len, hgnn_in, hgnn_hidden, hgnn_classes, transformer_dim=64, fusion_dim=64, num_classes=2):
        super(FusionGeneExpressionModel, self).__init__()
        self.hgnn = DNA_HGNN(in_ch=hgnn_in, hidden_ch=hgnn_hidden, num_classes=hgnn_classes)
        self.transformer = DiffusionTransformer(seq_len=seq_len, dim=transformer_dim, num_classes=num_classes)

        self.hgnn_proj = nn.Linear(hgnn_classes, fusion_dim)
        self.transformer_proj = nn.Linear(transformer_dim, fusion_dim)

        self.fusion = AttentionFusion(fusion_dim)  # use LightweightAttentionFusion if needed
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_seq, x_graph, G, timesteps):
        trans_feat = self.transformer(x_seq, timesteps)  # (B, transformer_dim)
        hgnn_feat = self.hgnn(x_graph, G)  # (B, hgnn_classes)

        trans_feat = self.transformer_proj(trans_feat)  # (B, fusion_dim)
        hgnn_feat = self.hgnn_proj(hgnn_feat).mean(dim=1)   # (B, fusion_dim)

        fused = self.fusion(hgnn_feat, trans_feat)  # (B, fusion_dim)
        return self.classifier(fused)  # (B, num_classes)


import numpy as np

seq_data = np.load("dataset/dataset1/Original_data/seq.npy")[:2000]
labels=np.load("dataset/dataset1/Original_data/train_labels.npy")[:2000]
graph_feat = seq_data.reshape((seq_data.shape[0], -1))
feature = graph_feat.shape[1]
G = torch.eye(feature)
from sklearn.metrics.pairwise import cosine_similarity
G = cosine_similarity(graph_feat.T)

import torch
from torch.utils.data import Dataset, DataLoader


class GeneExpressionDataset(Dataset):
    def __init__(self, seq_data, graph_feat, labels, global_G, node_indices_per_sample):
        self.seq_data = seq_data  # shape (N, 6000, 4)
        self.graph_feat = graph_feat  # shape (N, F)
        self.labels = labels  # shape (N,)
        self.global_G = global_G  # shape (24000, 24000)
        self.node_indices_per_sample = node_indices_per_sample  # list of arrays of node indices per sample

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seq_data[idx], dtype=torch.float32)
        graph_feat = torch.tensor(self.graph_feat[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Extract smaller adjacency matrix for this sample
        node_idx = self.node_indices_per_sample[idx]
        G_small = self.global_G[np.ix_(node_idx, node_idx)]
        G_small = torch.tensor(G_small, dtype=torch.float32)

        return seq, graph_feat, label, G_small



from torch.utils.data import DataLoader

def collate_fn(batch):
    seqs, graph_feats, labels, G_smalls = zip(*batch)
    seqs = torch.stack(seqs)
    graph_feats = torch.stack(graph_feats)
    labels = torch.stack(labels)
    G_smalls = torch.stack(G_smalls)
    return seqs, graph_feats, labels, G_smalls
N = len(labels)

# And each sample corresponds to 4 nodes, just an example:
num_nodes_per_sample = 4

# For simplicity, create dummy node indices for each sample, e.g., non-overlapping slices:
node_indices_per_sample = []

for i in range(N):
    start = i * num_nodes_per_sample
    end = start + num_nodes_per_sample
    node_indices_per_sample.append(list(range(start, end)))

train_dataset = GeneExpressionDataset(seq_data, graph_feat, labels, G, node_indices_per_sample)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)



model = FusionGeneExpressionModel(
    seq_len=6000,
    hgnn_in=graph_feat.shape[1],   # input features to HGNN (F)
    hgnn_hidden=128,
    hgnn_classes=128,
    transformer_dim=128,
    num_classes=3
)


import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for x_seq, x_graph, y, G_small in train_loader:
        x_seq = x_seq.to(device)
        x_graph = x_graph.to(device)
        y = y.to(device)
        G_small = G_small.to(device)  # pass the batch adjacency here!

        optimizer.zero_grad()
        timesteps = torch.randint(0, 1000, (x_seq.size(0),), dtype=torch.long).to(device)

        outputs = model(x_seq, x_graph, G_small, timesteps)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")
