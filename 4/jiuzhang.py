"""
九章算术 语言模型 — word2vec + LSTM

"""

import os
import sys
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- Windows 控制台 UTF-8 编码 ----
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---- matplotlib 中文字体设置 ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 全局设置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = '九章算经'
MODEL_DIR = 'output'
os.makedirs(MODEL_DIR, exist_ok=True)

print(f'Device: {DEVICE}')

# ============================================================
# 1. 数据加载与预处理
# ============================================================

def load_text(filepath):
    """加载 gb18030 编码的九章算术文本并清洗"""
    with open(filepath, 'rb') as f:
        raw = f.read()

    # 尝试多种中文编码
    for enc in ['gb18030', 'gbk', 'utf-8', 'big5']:
        try:
            text = raw.decode(enc)
            if any('一' <= c <= '鿿' for c in text[:500]):
                print(f'编码检测: {enc}')
                break
        except:
            continue

    # 清洗：去掉现代校注标记、页码等，保留正文
    # 去掉方括号注释 [……]
    text = re.sub(r'\[.+?\]', '', text)
    # 去掉连续空格和空行
    text = re.sub(r'[ \t]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去掉纯标点/数字行
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) >= 2]
    text = '\n'.join(lines)

    print(f'清洗后总字符数: {len(text)}')
    print(f'唯一字符数: {len(set(text))}')
    return text


def build_vocab(text, min_freq=2):
    """构建字符级词表"""
    counter = Counter(text)
    chars = ['<PAD>', '<UNK>'] + [c for c, f in counter.most_common() if f >= min_freq]
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    print(f'词表大小: {len(chars)} (min_freq={min_freq})')
    return char2idx, idx2char


# ============================================================
# 2. Word2Vec 词嵌入训练
# ============================================================

def train_word2vec(text, char2idx, embed_dim=128):
    """使用 gensim 训练 word2vec，并以句子为单位"""
    from gensim.models import Word2Vec

    # 按句号、问号等切分为"句子"
    sentences = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 按标点切分
        for seg in re.split(r'[。；？\n]', line):
            seg = seg.strip()
            if len(seg) >= 2:
                sentences.append(list(seg))

    print(f'句子数: {len(sentences)}')
    print(f'示例句子: {"".join(sentences[20])}')

    model = Word2Vec(
        sentences,
        vector_size=embed_dim,
        window=5,
        min_count=2,
        workers=4,
        sg=1,          
        epochs=30,
    )

    # 构建嵌入矩阵（对齐到 char2idx）
    vocab_size = len(char2idx)
    embed_matrix = np.random.normal(scale=0.01, size=(vocab_size, embed_dim)).astype(np.float32)

    hit = 0
    for char, idx in char2idx.items():
        if char in model.wv:
            embed_matrix[idx] = model.wv[char]
            hit += 1

    print(f'Word2Vec 覆盖率: {hit}/{vocab_size} ({100*hit/vocab_size:.1f}%)')
    return model, embed_matrix


def analyze_embeddings(w2v_model, char2idx, idx2char, embed_matrix):
    """语义探查：t-SNE 可视化 + 词类比 + 相似词检索"""
    print('\n' + '=' * 60)
    print('【语义探查 — 验证模型是否学到九章知识】')
    print('=' * 60)

    # ---- 2a. t-SNE 可视化 ----
    # 选取九章中的核心数学概念字
    target_groups = {
        '面积相关': '田积广从步亩里顷',
        '体积相关': '商功体积立方堑堵',
        '分数运算': '分约乘减退加母实法',
        '方程相关': '程方程正负如盈不足',
        '比例相关': '粟米衰分斛斗升石',
        '勾股相关': '勾股弦邪方幂',
        '通用量词': '一二三四五六七八九十百千万',
    }

    all_chars = []
    all_labels = []
    color_map = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#7f8c8d']

    for i, (label, chars) in enumerate(target_groups.items()):
        for c in chars:
            if c in char2idx:
                all_chars.append(c)
                all_labels.append(i)

    vectors = embed_matrix[[char2idx[c] for c in all_chars]]
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=5)
    vec_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 9))
    for i, (label, _) in enumerate(target_groups.items()):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            plt.scatter(vec_2d[mask, 0], vec_2d[mask, 1],
                        c=color_map[i], label=label, s=80, alpha=0.8)
            for j in np.where(mask)[0]:
                plt.annotate(all_chars[j], (vec_2d[j, 0], vec_2d[j, 1]),
                             fontsize=11,
                             xytext=(3, 3), textcoords='offset points')

    plt.title('九章算术 核心概念字 t-SNE 可视化', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'tsne_embeddings.png'), dpi=150)
    plt.close()
    print(f'[OK] t-SNE 图已保存至 {MODEL_DIR}/tsne_embeddings.png')

    # ---- 2b. 词类比测试 ----
    print('\n--- 词类比测试 ---')
    analogies = [
        ('方', '田', '商', '功'),    # 方田:面积 → 商功:体积
        ('广', '积', '深', '积'),    # 广→积 类比 深→积
        ('乘', '分', '加', '分'),    # 乘→分 类比 加→分
        ('一', '十', '十', '百'),    # 进位
    ]
    for a, b, c, expected_d in analogies:
        if all(x in w2v_model.wv for x in [a, b, c]):
            result = w2v_model.wv.most_similar(positive=[c, b], negative=[a], topn=5)
            result_str = ', '.join([f'{w}({s:.3f})' for w, s in result])
            hit = 'OK' if expected_d in [w for w, _ in result[:3]] else ' -'
            print(f'  {hit} {a}→{b} :: {c}→? 期望={expected_d}  top5: [{result_str}]')

    # ---- 2c. 相似词检索 ----
    print('\n--- 核心术语相似词 ---')
    for word in ['田', '积', '程', '分', '勾', '斛']:
        if word in w2v_model.wv:
            sim = w2v_model.wv.most_similar(word, topn=8)
            sim_str = ', '.join([f'{w}({s:.3f})' for w, s in sim])
            print(f'  {word}: [{sim_str}]')

    # ---- 2d. 余弦相似度矩阵热力图 ----
    key_terms = ['田', '积', '商', '功', '分', '程', '勾', '股', '斛', '斗']
    key_terms = [t for t in key_terms if t in char2idx]
    key_vecs = embed_matrix[[char2idx[t] for t in key_terms]]
    key_vecs = key_vecs / (np.linalg.norm(key_vecs, axis=1, keepdims=True) + 1e-8)
    sim_mat = cosine_similarity(key_vecs)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_mat, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(key_terms)))
    ax.set_yticks(range(len(key_terms)))
    ax.set_xticklabels(key_terms, fontsize=12)
    ax.set_yticklabels(key_terms, fontsize=12)
    for i in range(len(key_terms)):
        for j in range(len(key_terms)):
            ax.text(j, i, f'{sim_mat[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.title('九章算术 核心术语余弦相似度', fontsize=14)
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'similarity_heatmap.png'), dpi=150)
    plt.close()
    print(f'[OK] 相似度热力图已保存至 {MODEL_DIR}/similarity_heatmap.png')


# ============================================================
# 3. LSTM 语言模型
# ============================================================

class CharDataset(Dataset):
    """字符级序列数据集"""
    def __init__(self, text, char2idx, seq_len=50):
        self.seq_len = seq_len
        self.data = [char2idx.get(c, char2idx['<UNK>']) for c in text]
        self.data = [d for d in self.data if d != char2idx['<PAD>']]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.5,
                 pretrained_embed=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embed is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
            if freeze_embed:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size):
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(DEVICE)
        c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(DEVICE)
        return (h0, c0)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(dataloader.batch_size)

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # 每个 batch 重新初始化 hidden state（避免跨 batch 梯度传播）
        hidden = tuple(h.detach() for h in hidden)

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(dataloader.batch_size)

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, hidden = model(x, hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, max_len=100, temperature=0.8):
    """根据 prompt 续写文本"""
    model.eval()
    vocab_size = len(char2idx)
    unk_idx = char2idx['<UNK>']

    chars = [char2idx.get(c, unk_idx) for c in prompt]
    x = torch.tensor([chars], dtype=torch.long).to(DEVICE)
    hidden = model.init_hidden(1)

    generated = list(prompt)

    for _ in range(max_len):
        logits, hidden = model(x[:, -1:], hidden)  # 只输入最后一个字符
        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)

        # 过滤掉特殊 token
        for bad_id in [char2idx.get('<PAD>', -1), char2idx.get('<UNK>', -1)]:
            if bad_id >= 0:
                probs[bad_id] = 0

        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx2char.get(next_idx, '？')

        generated.append(next_char)
        # 将新字符追加到序列中
        next_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
        x = torch.cat([x, next_tensor], dim=1)

        # 在自然断句处停止
        if next_char in '。\n' and len(generated) > len(prompt) + 10:
            if generated[-2] in '。\n':
                break

    return ''.join(generated)


# ============================================================
# 4. 主流程
# ============================================================

def main():
    # ---- 4a. 加载数据 ----
    filepath = os.path.join(DATA_DIR, '九章算经.txt')
    text = load_text(filepath)
    char2idx, idx2char = build_vocab(text, min_freq=1)

    # 划分数据集
    total_chars = len(text)
    train_end = int(total_chars * 0.85)
    val_end = int(total_chars * 0.92)

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    print(f'\n数据划分: 训练={len(train_text)}, 验证={len(val_text)}, 测试={len(test_text)}')

    # ---- 4b. 训练 Word2Vec ----
    print('\n' + '=' * 60)
    print('【训练 Word2Vec 词嵌入】')
    print('=' * 60)
    w2v_model, embed_matrix = train_word2vec(train_text, char2idx, embed_dim=128)

    # 保存 Word2Vec
    w2v_model.save(os.path.join(MODEL_DIR, 'word2vec.model'))
    np.save(os.path.join(MODEL_DIR, 'embed_matrix.npy'), embed_matrix)

    # ---- 4c. 语义探查 ----
    analyze_embeddings(w2v_model, char2idx, idx2char, embed_matrix)

    # ---- 4d. 训练 LSTM ----
    print('\n' + '=' * 60)
    print('【训练 LSTM 语言模型】')
    print('=' * 60)

    SEQ_LEN = 50
    BATCH_SIZE = 64
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LR = 0.002
    EPOCHS = 20

    train_dataset = CharDataset(train_text, char2idx, SEQ_LEN)
    val_dataset = CharDataset(val_text, char2idx, SEQ_LEN)
    test_dataset = CharDataset(test_text, char2idx, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    vocab_size = len(char2idx)
    model = LSTMLanguageModel(
        vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
        pretrained_embed=embed_matrix, freeze_embed=False
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')

    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_ppls = []
    val_ppls = []
    best_val_ppl = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_lstm.pt'))

        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{EPOCHS} | '
                  f'Train Loss: {train_loss:.4f} (ppl={train_ppl:.1f}) | '
                  f'Val Loss: {val_loss:.4f} (ppl={val_ppl:.1f}) | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.5f}')

    # ---- 4e. 训练曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_ppls, label='Train PPL', color='#3498db')
    axes[0].plot(val_ppls, label='Val PPL', color='#e74c3c')
    axes[0].axhline(y=len(char2idx), color='gray', linestyle='--', alpha=0.5,
                    label=f'Random baseline ({len(char2idx)})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Perplexity')
    axes[0].set_title('LSTM 语言模型 — 困惑度曲线')
    axes[0].legend()
    axes[0].set_yscale('log')

    axes[1].plot(train_ppls, label='Train PPL', color='#3498db')
    axes[1].plot(val_ppls, label='Val PPL', color='#e74c3c')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('困惑度（线性坐标）')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f'[OK] 训练曲线已保存至 {MODEL_DIR}/training_curves.png')

    # ---- 4f. 测试集评估 ----
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_lstm.pt')))
    test_loss = evaluate(model, test_loader, criterion)
    test_ppl = np.exp(test_loss)
    print(f'\n测试集 Perplexity: {test_ppl:.2f}')
    print(f'随机基线 Perplexity (词表大小): {vocab_size}')

    # ---- 4g. 条件生成演示 ----
    print('\n' + '=' * 60)
    print('【条件生成演示 — 模型是否学到了九章的问题结构？】')
    print('=' * 60)

    prompts = [
        '今有田广十五步，从十六步。',
        '今有粟一斗，',
        '今有堤，下广二丈，',
        '今有勾三尺，股四尺，',
        '今有积五万五千二百二十五步。',
    ]

    for i, prompt in enumerate(prompts):
        print(f'\n--- 示例 {i+1} ---')
        print(f'Prompt:    {prompt}')
        generated = generate(model, prompt, char2idx, idx2char, max_len=80, temperature=0.8)
        print(f'Generated: {generated}')

    print('\n' + '=' * 60)
    print('训练完成！所有输出文件保存在 output/ 目录中。')
    print('=' * 60)


if __name__ == '__main__':
    main()
