"""A lightweight, dependency-free learned selection heuristic.

Deliberately implemented in pure NumPy rather than PyTorch/PyTorch
Geometric: the graphs here are small (a single floor plan has maybe
15-40 rooms), the dataset is currently small, and this project has
already hit real friction from heavier dependencies (see the Bazzite
pip/venv issues earlier in this project). A hand-rolled two-layer MLP
gets the same core capability -- context-aware scoring instead of a
static per-tile weight -- without adding a new dependency beyond what's
already required (numpy).

This is NOT a full message-passing GNN (it doesn't do multi-hop graph
convolutions). It IS a legitimate context-aware scorer: given a
candidate room type and a summary of its immediate neighborhood (the
room types already placed around the target cell), it predicts how
plausible that placement is, trained on real room-adjacency examples via
extract_room_graph. If a genuine multi-layer GNN is wanted later, this
module's train/predict interface is the seam to swap it in at --
everything in wfc_core.py only ever calls `predict()`.
"""

import numpy as np


class ContextScorer:
    def __init__(self, room_types, hidden_dim=16, seed=0):
        self.room_types = list(room_types)
        self.type_to_idx = {rt: i for i, rt in enumerate(self.room_types)}
        self.n_types = len(self.room_types)
        # Input features: one-hot(candidate room type) + mean one-hot of
        # up to 4 neighboring room types (a simple permutation-invariant
        # neighborhood summary -- this is what stands in for message
        # passing here).
        self.input_dim = self.n_types * 2
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, size=(self.input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.3, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)

    def _one_hot(self, room_type):
        v = np.zeros(self.n_types)
        if room_type in self.type_to_idx:
            v[self.type_to_idx[room_type]] = 1.0
        return v

    def _features(self, candidate_room_type, neighbor_room_types):
        cand = self._one_hot(candidate_room_type)
        if neighbor_room_types:
            neigh = np.mean([self._one_hot(rt) for rt in neighbor_room_types], axis=0)
        else:
            neigh = np.zeros(self.n_types)
        return np.concatenate([cand, neigh])

    def _forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        out = 1.0 / (1.0 + np.exp(-z2))  # sigmoid
        return z1, a1, out

    def predict(self, candidate_room_type, neighbor_room_types):
        """Return a plausibility score in (0, 1) for placing a room of
        `candidate_room_type` next to the given list of already-placed
        neighboring room types. Higher = more plausible."""
        x = self._features(candidate_room_type, neighbor_room_types)
        _, _, out = self._forward(x)
        return float(out[0])

    def train(self, examples, epochs=200, lr=0.05, verbose=False):
        """examples: list of (candidate_room_type, neighbor_room_types, label)
        where label is 1.0 for a real/plausible adjacency and 0.0 for a
        negative (implausible) one. Trains via plain batch gradient
        descent -- no autograd, gradients derived by hand for this small
        two-layer network."""
        X = np.array([self._features(c, n) for c, n, _ in examples])
        y = np.array([[lbl] for _, _, lbl in examples])

        loss = None
        for epoch in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.W2 + self.b2
            out = 1.0 / (1.0 + np.exp(-z2))

            eps = 1e-9
            loss = -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))

            d_out = (out - y) / len(examples)
            dW2 = a1.T @ d_out
            db2 = d_out.sum(axis=0)
            d_a1 = d_out @ self.W2.T
            d_z1 = d_a1 * (1 - a1 ** 2)
            dW1 = X.T @ d_z1
            db1 = d_z1.sum(axis=0)

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            if verbose and epoch % 50 == 0:
                acc = np.mean((out > 0.5) == y)
                print(f"  epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.1%}")

        return loss

    def save(self, path):
        np.savez(
            path,
            room_types=np.array(self.room_types),
            hidden_dim=self.hidden_dim,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        obj = cls(room_types=list(data["room_types"]), hidden_dim=int(data["hidden_dim"]))
        obj.W1, obj.b1 = data["W1"], data["b1"]
        obj.W2, obj.b2 = data["W2"], data["b2"]
        return obj