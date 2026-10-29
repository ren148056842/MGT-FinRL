"""TensorFlow models of the v4 forecaster: MGT, its ablations, and five graph baselines.

All models map a window [B, P, N, F] to six outputs per stock [B, N, 6] (standardised
log ratios h1, l1, c1, h2, l2, c2).  Relation matrices are used as archived (order KD, QA, SH).

graph = 'mgnn' : MGT propagation H <- tanh(Dense(A_f H)), A_f = W_M + sum_r w_r alpha_r A_r
                 (W_M Glorot-uniform; alpha_r learnable or fixed, initialised at 1/rho(A_r),
                 rho = spectral radius, so each relation layer enters at unit scale)
        'gcn'  : fixed symmetric-normalised fused graph D^-1/2 (A_bar + I) D^-1/2,
                 A_bar = sum_r w_r max(A_r, 0) / max(A_r) with zero diagonal (no learnable fusion)
        'gat'  : single-head graph attention restricted to the edges of A_bar plus self-loops
        'identity': same layers as the graph branch applied to each stock separately (A = I);
                 controls for the capacity of the graph branch without cross-stock propagation
        None   : no graph branch
temporal = 'transformer' : one encoder block applied to each stock along time, learned positional
                           embedding, 4 heads, feed-forward width 128, dropout 0.1
           'gru'         : GRU with 64 units applied to each stock along time
           None          : last time step only
The graph branch output (widths 16, 32, 64) is concatenated with the inputs, as in MGT.
"""
import numpy as np
import tensorflow as tf

UNITS_GNN = [16, 32, 64]

CONFIGS = {
    'A0_Full_MGT':         dict(graph='mgnn', w=[1, 1, 1], learn_alpha=True,  temporal='transformer'),
    'A1_Transformer_only': dict(graph=None,   w=[0, 0, 0], learn_alpha=False, temporal='transformer'),
    'A2_MGNN_only':        dict(graph='mgnn', w=[1, 1, 1], learn_alpha=True,  temporal=None),
    'A3_SH_only':          dict(graph='mgnn', w=[0, 0, 1], learn_alpha=True,  temporal='transformer'),
    'A4_QA_only':          dict(graph='mgnn', w=[0, 1, 0], learn_alpha=True,  temporal='transformer'),
    'A5_KD_only':          dict(graph='mgnn', w=[1, 0, 0], learn_alpha=True,  temporal='transformer'),
    'A6_SH_QA':            dict(graph='mgnn', w=[0, 1, 1], learn_alpha=True,  temporal='transformer'),
    'A7_SH_KD':            dict(graph='mgnn', w=[1, 0, 1], learn_alpha=True,  temporal='transformer'),
    'A8_QA_KD':            dict(graph='mgnn', w=[1, 1, 0], learn_alpha=True,  temporal='transformer'),
    'A9_Fixed_fusion':     dict(graph='mgnn', w=[1, 1, 1], learn_alpha=False, temporal='transformer'),
    'B1_GCN_Transformer':  dict(graph='gcn',  w=[1, 1, 1], learn_alpha=False, temporal='transformer'),
    'B2_GAT_Transformer':  dict(graph='gat',  w=[1, 1, 1], learn_alpha=False, temporal='transformer'),
    'B3_GNN_GRU':          dict(graph='mgnn', w=[1, 1, 1], learn_alpha=True,  temporal='gru'),
    'B4_GCN_GRU':          dict(graph='gcn',  w=[1, 1, 1], learn_alpha=False, temporal='gru'),
    'B5_GAT_GRU':          dict(graph='gat',  w=[1, 1, 1], learn_alpha=False, temporal='gru'),
    'A10_No_cross_stock':  dict(graph='identity', w=[0, 0, 0], learn_alpha=False, temporal='transformer'),
    # initialisation sensitivity of the fusion scalars (A0 otherwise unchanged)
    'I1_alpha_one':        dict(graph='mgnn', w=[1, 1, 1], learn_alpha=True, temporal='transformer', alpha_init='one'),
    'I2_alpha_small':      dict(graph='mgnn', w=[1, 1, 1], learn_alpha=True, temporal='transformer', alpha_scale=0.1),
    'I3_alpha_large':      dict(graph='mgnn', w=[1, 1, 1], learn_alpha=True, temporal='transformer', alpha_scale=10.0),
}
ALPHA_VARIANTS = ['I1_alpha_one', 'I2_alpha_small', 'I3_alpha_large']
RL_VARIANTS = ['A0_Full_MGT', 'A1_Transformer_only', 'A2_MGNN_only', 'A6_SH_QA', 'A7_SH_KD', 'A8_QA_KD', 'A9_Fixed_fusion']


def spectral_radius(A):
    return float(np.abs(np.linalg.eigvals(np.asarray(A, float))).max())


def fused_static(graphs, w):
    N = graphs[0].shape[0]
    A = np.zeros((N, N))
    for g, wi in zip(graphs, w):
        if wi:
            gp = np.maximum(np.asarray(g, float), 0)
            np.fill_diagonal(gp, 0)
            if gp.max() > 0:
                A += gp / gp.max()
    return A


def gcn_matrix(graphs, w):
    A = fused_static(graphs, w) + np.eye(graphs[0].shape[0])
    d = A.sum(1)
    Dm = np.diag(1 / np.sqrt(d))
    return (Dm @ A @ Dm).astype(np.float32)


class GATLayer(tf.keras.layers.Layer):
    def __init__(self, units, mask, **kw):
        super().__init__(**kw)
        self.units = units
        self.mask = tf.constant(mask, dtype=tf.bool)

    def build(self, shape):
        self.W = self.add_weight('W', shape=(int(shape[-1]), self.units), initializer='glorot_uniform')
        self.a1 = self.add_weight('a1', shape=(self.units, 1), initializer='glorot_uniform')
        self.a2 = self.add_weight('a2', shape=(self.units, 1), initializer='glorot_uniform')
        self.b = self.add_weight('b', shape=(self.units,), initializer='zeros')

    def call(self, h):                                     # h: [B, P, N, F]
        z = tf.einsum('bpnf,fu->bpnu', h, self.W)
        f1 = tf.einsum('bpnu,uk->bpnk', z, self.a1)                     # [B, P, N, 1]
        f2 = tf.einsum('bpnu,uk->bpkn', z, self.a2)                     # [B, P, 1, N]
        e = tf.nn.leaky_relu(f1 + f2, 0.2)                                # e_ij = a1.z_i + a2.z_j
        e = tf.where(self.mask, e, tf.fill(tf.shape(e), -1e9))
        att = tf.nn.softmax(e, axis=-1)                    # [B, P, N, N]
        return tf.nn.elu(tf.matmul(att, z) + self.b)


class Forecaster(tf.keras.Model):
    def __init__(self, N, F, P, graphs, cfg, n_out=6, num_heads=4, dff=128, dropout=0.1, gru_units=64):
        super().__init__()
        self.cfg = dict(cfg)
        self.graph = cfg['graph']
        self.temporal = cfg['temporal']
        w = [float(x) for x in cfg['w']]
        if self.graph == 'mgnn':
            self.fixed = [tf.constant(g, tf.float32) for g in graphs]
            self.w = tf.constant(w, tf.float32)
            init = np.array([1.0 / spectral_radius(g) if spectral_radius(g) > 0 else 0.0 for g in graphs], np.float32)
            init = init * float(cfg.get('alpha_scale', 1.0))
            if cfg.get('alpha_init') == 'one':
                init = np.ones(len(graphs), np.float32)
            self.WM = self.add_weight(name='base_adjacency', shape=(N, N), initializer='glorot_uniform', trainable=True)
            self.alpha = self.add_weight(name='relation_alpha', shape=(len(graphs),),
                                         initializer=tf.keras.initializers.Constant(init), trainable=cfg['learn_alpha'])
            self.gnn = [tf.keras.layers.Dense(u, activation='tanh') for u in UNITS_GNN]
        elif self.graph == 'gcn':
            self.Ahat = tf.constant(gcn_matrix(graphs, w))
            self.gnn = [tf.keras.layers.Dense(u, activation='tanh') for u in UNITS_GNN]
        elif self.graph == 'gat':
            mask = (fused_static(graphs, w) > 0) | np.eye(N, dtype=bool)
            self.gnn = [GATLayer(u, mask) for u in UNITS_GNN]
        elif self.graph == 'identity':
            self.gnn = [tf.keras.layers.Dense(u, activation='tanh') for u in UNITS_GNN]
        D = F + (UNITS_GNN[-1] if self.graph else 0)
        self.D = D
        if self.temporal == 'transformer':
            self.pos = self.add_weight(name='pos_embedding', shape=(P, D), initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
            self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(1, D // num_heads))
            self.n1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.n2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(D)])
            self.drop = tf.keras.layers.Dropout(dropout)
        elif self.temporal == 'gru':
            self.rnn = tf.keras.layers.GRU(gru_units)
        self.out = tf.keras.layers.Dense(n_out)

    def adjacency(self):
        A = self.WM
        for i, M in enumerate(self.fixed):
            A = A + self.w[i] * self.alpha[i] * M
        return A

    def graph_branch(self, x):
        h = x
        if self.graph == 'mgnn':
            A = self.adjacency()
            for layer in self.gnn:
                h = layer(tf.einsum('ij,bpjf->bpif', A, h))
        elif self.graph == 'gcn':
            for layer in self.gnn:
                h = layer(tf.einsum('ij,bpjf->bpif', self.Ahat, h))
        else:
            for layer in self.gnn:
                h = layer(h)
        return h

    def call(self, x, training=False):
        if self.graph:
            x = tf.concat([x, self.graph_branch(x)], axis=-1)
        B, P, N = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        if self.temporal is None:
            z = x[:, -1]                                        # [B, N, D]
        else:
            s = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B * N, P, self.D])
            if self.temporal == 'transformer':
                s = s + self.pos
                a = self.attn(s, s, s, training=training)
                s = self.n1(s + self.drop(a, training=training))
                s = self.n2(s + self.drop(self.ffn(s), training=training))
                z = s[:, -1]
            else:
                z = self.rnn(s)
            z = tf.reshape(z, [B, N, -1])
        return self.out(z)


def masked_mse(y, yhat, m):
    m = tf.cast(m, tf.float32)
    return tf.reduce_sum(m * tf.square(y - yhat)) / tf.maximum(tf.reduce_sum(m), 1.0)
