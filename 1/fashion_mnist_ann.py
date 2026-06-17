#!/usr/bin/python
# 基于 demo_mnist_ann.py 改编，用于 FashionMNIST 数据集

import numpy as np
import struct
import time

# ==================== 激活函数 ====================

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# ==================== 激活函数导数 ====================

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


class NeuralNetwork:
    def __init__(self, layers, activation='relu', opt_alg='SGD'):
        """
        layers:   网络层结构，如 [784, 128, 10]
        activation: 激活函数 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu'
        opt_alg:   优化器 'GD' | 'SGD' | 'ADAM'
        """
        # 激活函数映射
        act_map = {
            'sigmoid':    (sigmoid,    sigmoid_deriv),
            'tanh':       (tanh,       tanh_deriv),
            'relu':       (relu,       relu_deriv),
            'leaky_relu': (leaky_relu, leaky_relu_deriv),
        }
        if activation not in act_map:
            raise ValueError(f"不支持的激活函数: {activation}")
        self.activation, self.activation_deriv = act_map[activation]
        self.activation_name = activation

        # 优化器映射
        opt_map = {'GD': self._gd, 'SGD': self._sgd, 'ADAM': self._adam}
        if opt_alg not in opt_map:
            raise ValueError(f"不支持的优化器: {opt_alg}")
        self.opt = opt_map[opt_alg]
        self.opt_name = opt_alg

        # 初始化 Adam 参数
        if opt_alg == 'ADAM':
            self.mw, self.vw = [], []
            self.mtheta, self.vtheta = [], []
            for i in range(len(layers) - 1):
                self.mw.append(np.zeros((layers[i + 1], layers[i])))
                self.vw.append(np.zeros((layers[i + 1], layers[i])))
                self.mtheta.append(np.zeros((layers[i + 1], 1)))
                self.vtheta.append(np.zeros((layers[i + 1], 1)))

        # 初始化权重和偏置  (-1, 1)
        self.weights, self.thetas = [], []
        rng = np.random.default_rng(42)
        for i in range(len(layers) - 1):
            # He 初始化 (适合 ReLU 族) 或 Xavier (适合 sigmoid/tanh)
            if activation in ('relu', 'leaky_relu'):
                scale = np.sqrt(2.0 / layers[i])
            else:
                scale = np.sqrt(1.0 / layers[i])
            self.weights.append(rng.uniform(-scale, scale, (layers[i + 1], layers[i])))
            self.thetas.append(np.zeros((layers[i + 1], 1)))

        self.layers = layers

    # ---------- 前向传播 ----------
    def propagation(self, x, k=1.0):
        temp = x
        for w, theta in zip(self.weights, self.thetas):
            temp = self.activation(np.dot(w, temp) + theta)
        return k * temp

    # ---------- 反向传播 ----------
    def backpropagation(self, x, error):
        n_w = len(self.weights)
        z = []
        K = []

        for i in range(n_w):
            if i == 0:
                inp = x
            else:
                inp = self.activation(z[i - 1])
            z.append(np.dot(self.weights[i], inp) + self.thetas[i])
            K.append(inp)

        delta = [None] * n_w
        for i in range(n_w - 1, -1, -1):
            if i == n_w - 1:
                delta[i] = error * self.activation_deriv(z[i])
            else:
                delta[i] = np.dot(self.weights[i + 1].T, delta[i + 1]) * self.activation_deriv(z[i])

        dweights = [np.dot(d, k.T) for d, k in zip(delta, K)]
        dthetas = delta
        return dweights, dthetas

    # ---------- 批量梯度下降 (向量化) ----------
    def _gd(self, X, Y, k, lr, epoch):
        n = X.shape[1]
        n_w = len(self.weights)

        # 前向 — 全批量矩阵运算
        Z = []
        A = [X]
        for i in range(n_w):
            Z.append(np.dot(self.weights[i], A[i]) + self.thetas[i])
            A.append(self.activation(Z[i]))
        output = k * A[-1]

        error = Y - output
        perf = 0.5 * np.sum(error ** 2)

        # 反向 — 存储每层 delta
        delta = [None] * n_w
        for i in range(n_w - 1, -1, -1):
            if i == n_w - 1:
                delta[i] = error * self.activation_deriv(Z[i])
            else:
                delta[i] = (np.dot(self.weights[i + 1].T, delta[i + 1])
                            * self.activation_deriv(Z[i]))

        # 更新
        for i in range(n_w):
            self.weights[i] += k * lr / n * np.dot(delta[i], A[i].T)
            self.thetas[i]  += k * lr / n * np.sum(delta[i], axis=1, keepdims=True)

        return perf

    # ---------- 随机梯度下降 (mini-batch 1%) ----------
    def _sgd(self, X, Y, k, lr, _epoch):
        ddw = [np.zeros_like(w) for w in self.weights]
        ddt = [np.zeros_like(t) for t in self.thetas]
        perf = 0.0
        total = X.shape[1]
        idx = np.random.permutation(total)
        batch = max(1, total // 100)  # 1% 样本

        for j in range(batch):
            inp = X[:, idx[j]].reshape(-1, 1)
            y   = Y[:, idx[j]].reshape(-1, 1)
            out = self.propagation(inp, k)
            err = y - out
            perf += 0.5 * np.sum(err ** 2)
            dw, dt = self.backpropagation(inp, err)
            for i in range(len(self.weights)):
                ddw[i] += dw[i]
                ddt[i] += dt[i]

        for i in range(len(self.weights)):
            self.weights[i] += k * lr / batch * ddw[i]
            self.thetas[i]  += k * lr / batch * ddt[i]
        return perf

    # ---------- Adam 优化器 (mini-batch 10%) ----------
    def _adam(self, X, Y, k, lr, epoch):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        ddw = [np.zeros_like(w) for w in self.weights]
        ddt = [np.zeros_like(t) for t in self.thetas]
        perf = 0.0
        total = X.shape[1]
        idx = np.random.permutation(total)
        batch = max(1, total // 10)

        for j in range(batch):
            inp = X[:, idx[j]].reshape(-1, 1)
            y   = Y[:, idx[j]].reshape(-1, 1)
            out = self.propagation(inp, k)
            err = y - out
            perf += 0.5 * np.sum(err ** 2)
            dw, dt = self.backpropagation(inp, err)
            for i in range(len(self.weights)):
                ddw[i] += dw[i]
                ddt[i] += dt[i]

        for j in range(len(self.weights)):
            ddw[j] *= k / batch
            ddt[j] *= k / batch

            self.mw[j] = beta1 * self.mw[j] + (1 - beta1) * ddw[j]
            self.vw[j] = beta2 * self.vw[j] + (1 - beta2) * ddw[j] ** 2
            self.mtheta[j] = beta1 * self.mtheta[j] + (1 - beta1) * ddt[j]
            self.vtheta[j] = beta2 * self.vtheta[j] + (1 - beta2) * ddt[j] ** 2

            mw_hat = self.mw[j] / (1 - beta1 ** epoch)
            vw_hat = self.vw[j] / (1 - beta2 ** epoch)
            mt_hat = self.mtheta[j] / (1 - beta1 ** epoch)
            vt_hat = self.vtheta[j] / (1 - beta2 ** epoch)

            self.weights[j] += lr * mw_hat / (np.sqrt(vw_hat) + eps)
            self.thetas[j]  += lr * mt_hat / (np.sqrt(vt_hat) + eps)
        return perf

    # ---------- 训练 ----------
    def train(self, X, Y, X_test=None, Y_test=None,
              k=1.0, learning_rate=0.01, epochs=100):
        print(f"网络结构: {self.layers}")
        print(f"激活函数: {self.activation_name}")
        print(f"优化器:   {self.opt_name}")
        print(f"学习率:   {learning_rate}")
        print(f"训练轮数: {epochs}")
        print(f"训练样本: {X.shape[1]}, 测试样本: {X_test.shape[1] if X_test is not None else 0}")
        print("-" * 50)

        t_start = time.time()
        for ep in range(epochs):
            perf = self.opt(X, Y, k, learning_rate, ep + 1)

            # 训练集准确率
            pred = self.propagation(X, 1.0)
            train_acc = np.mean(np.argmax(Y, 0) == np.argmax(pred, 0))

            # 测试集准确率
            test_str = ""
            if X_test is not None and Y_test is not None:
                pred_t = self.propagation(X_test, 1.0)
                test_acc = np.mean(np.argmax(Y_test, 0) == np.argmax(pred_t, 0))
                test_str = f" | test_acc: {test_acc:.4f}"

            print(f"epoch {ep + 1:4d}/{epochs}  "
                  f"loss: {perf:.2f}  "
                  f"train_acc: {train_acc:.4f}{test_str}")

        print(f"训练用时: {time.time() - t_start:.1f}s")


# ==================== 数据加载 ====================

def load_fashion_mnist(img_path, lbl_path, num_train=8000):
    """
    加载 FashionMNIST 数据 (IDX 格式)
    num_train: 用于训练的数量，剩余作为测试
    """
    # 读取图像
    with open(img_path, 'rb') as f:
        buf = f.read()
    idx = 0
    magic, total, rows, cols = struct.unpack_from('>IIII', buf, idx)
    idx += 16
    pix = rows * cols  # 784

    X_train = np.zeros((pix, num_train))
    for i in range(num_train):
        X_train[:, i] = struct.unpack_from(f'>{pix}B', buf, idx)
        idx += pix

    X_test = np.zeros((pix, total - num_train))
    for i in range(total - num_train):
        X_test[:, i] = struct.unpack_from(f'>{pix}B', buf, idx)
        idx += pix

    # 读取标签
    with open(lbl_path, 'rb') as f:
        buf = f.read()
    idx = 8  # 跳过 magic + num

    Y_train = np.zeros((10, num_train))
    for i in range(num_train):
        label = struct.unpack_from('>B', buf, idx)[0]
        Y_train[label, i] = 1.0
        idx += 1

    Y_test = np.zeros((10, total - num_train))
    for i in range(total - num_train):
        label = struct.unpack_from('>B', buf, idx)[0]
        Y_test[label, i] = 1.0
        idx += 1

    # 归一化到 [0, 1]，避免数值溢出
    X_train = X_train.astype(np.float64) / 255.0
    X_test  = X_test.astype(np.float64)  / 255.0
    return X_train, Y_train, X_test, Y_test


# ==================== 主程序 ====================

if __name__ == "__main__":
    import os

    # 数据集路径
    base = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base, 'data', 'datasets', 't10k-images-idx3-ubyte')
    lbl_path = os.path.join(base, 'data', 'datasets', 't10k-labels-idx1-ubyte')

    # ---------- 可调参数 ----------
    NUM_TRAIN = 8000          # 训练样本数 (总共 10000)
    HIDDEN     = 128           # 隐藏层大小
    LAYERS      = [784, HIDDEN, 10]   # 网络结构
    ACTIVATION  = 'relu'       # 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu'
    OPTIMIZER   = 'ADAM'       # 'GD' | 'SGD' | 'ADAM'
    LEARNING_RATE = 0.001
    EPOCHS      = 30
    # ---------------------------

    # 加载数据
    print("正在加载 FashionMNIST 数据...")
    X_train, Y_train, X_test, Y_test = load_fashion_mnist(img_path, lbl_path, NUM_TRAIN)
    print(f"数据加载完成: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"               X_test  {X_test.shape},  Y_test  {Y_test.shape}\n")

    # 创建并训练网络
    net = NeuralNetwork(LAYERS, activation=ACTIVATION, opt_alg=OPTIMIZER)
    net.train(X_train, Y_train, X_test, Y_test,
              k=1.0, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    # 最终测试
    output = net.propagation(X_test, 1.0)
    final_acc = np.mean(np.argmax(Y_test, 0) == np.argmax(output, 0))
    print(f"\n最终测试准确率: {final_acc:.4f}")
