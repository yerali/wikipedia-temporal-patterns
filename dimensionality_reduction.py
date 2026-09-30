"""
Dimensionality reduction methods compared in the Supplementary Information.

Each function takes a matrix of shape (11 editions, n_features) and returns
the reduced representation on which Ward clustering is then applied:

    none         the matrix unchanged (reference)
    pca          principal component analysis, 5 components
    autoencoder  fully connected autoencoder, bottleneck of 10 neurons
    tsne         t-SNE, 2 components, perplexity 5, PCA initialisation
    umap         UMAP, 3 components, n_neighbors 3, min_dist 0.1

All stochastic methods use the fixed seed RANDOM_STATE.

TensorFlow and umap-learn are optional: if one of them is missing, the
corresponding function returns None and the method is skipped.

This module is imported by the Supplementary Information script and is
not meant to be run on its own.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    HAS_TF = True
except ImportError:
    HAS_TF = False


# =================================================================
# Parameters
# =================================================================

RANDOM_STATE = 42

PCA_COMPONENTS = 5
AE_ENCODING_DIM, AE_EPOCHS, AE_BATCH, AE_LR = 10, 5000, 256, 0.001
TSNE_PERPLEXITY, TSNE_N_COMPONENTS, TSNE_INIT = 5, 2, "pca"
UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS = 3, 0.1, 3


# =================================================================
# Preprocessing
# =================================================================

def global_scale(X):
    """
    Multiply the whole matrix by a single scalar so that its root mean
    square is one.

    After normalisation the cells are of order 1e-3, and the autoencoder
    may fail to learn on values that small. Since the factor is the same
    for every edition, it leaves all relative distances unchanged: Ward's
    method returns the same tree (only the heights are rescaled).
    """
    rms = np.sqrt(np.mean(X ** 2))
    return X if rms == 0 else X / rms


# =================================================================
# Reduction methods
# =================================================================

def reduce_none(X):
    """No reduction: the reference clustering."""
    return X


def reduce_pca(X):
    """Projection onto the first PCA_COMPONENTS principal components."""
    return PCA(n_components=PCA_COMPONENTS,
               random_state=RANDOM_STATE).fit_transform(X)


def reduce_autoencoder(X):
    """
    Encoding of a fully connected autoencoder with layers
    input-30-15-15-10-15-15-30-output and Leaky ReLU activations.

    Returns None if TensorFlow is not installed.
    """
    if not HAS_TF:
        return None
    tf.random.set_seed(RANDOM_STATE)
    d = X.shape[1]

    # Encoder.
    inp = Input(shape=(d,), name="input_layer")
    h = Dense(30, activation=tf.nn.leaky_relu)(inp)
    h = Dense(15, activation=tf.nn.leaky_relu)(h)
    h = Dense(15, activation=tf.nn.leaky_relu)(h)
    enc = Dense(AE_ENCODING_DIM, activation=tf.nn.leaky_relu)(h)

    # Decoder, symmetric to the encoder.
    h = Dense(15, activation=tf.nn.leaky_relu)(enc)
    h = Dense(15, activation=tf.nn.leaky_relu)(h)
    h = Dense(30, activation=tf.nn.leaky_relu)(h)
    out = Dense(d, activation=tf.nn.leaky_relu)(h)

    # Train the full autoencoder to reproduce its input.
    ae = Model(inputs=inp, outputs=out)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=AE_LR),
               loss="mean_squared_error")
    hist = ae.fit(X, X, epochs=AE_EPOCHS, batch_size=AE_BATCH,
                  shuffle=True, verbose=0)

    # Report how much the loss decreased during training.
    l0, l1 = hist.history["loss"][0], hist.history["loss"][-1]
    print(f"      autoencoder loss {l0:.3e} -> {l1:.3e} "
          f"(reduction {l0 / max(l1, 1e-30):.1f}x)")

    # Return the bottleneck representation.
    return Model(inputs=inp, outputs=enc).predict(X, verbose=0)


def reduce_tsne(X):
    """t-SNE embedding. The perplexity must be smaller than the number of
    samples, so it is capped at n_samples - 1."""
    perplexity = min(TSNE_PERPLEXITY, X.shape[0] - 1)
    return TSNE(n_components=TSNE_N_COMPONENTS, perplexity=perplexity,
                init=TSNE_INIT, random_state=RANDOM_STATE,
                learning_rate="auto").fit_transform(X)


def reduce_umap(X):
    """UMAP embedding. The number of neighbours must be smaller than the
    number of samples. Returns None if umap-learn is not installed."""
    if not HAS_UMAP:
        return None
    n_neighbors = min(UMAP_N_NEIGHBORS, X.shape[0] - 1)
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=UMAP_MIN_DIST,
                     n_components=UMAP_N_COMPONENTS,
                     random_state=RANDOM_STATE).fit_transform(X)


# Methods by name, and the order in which they are shown.
REDUCTIONS = {"none": reduce_none, "pca": reduce_pca,
              "autoencoder": reduce_autoencoder, "tsne": reduce_tsne,
              "umap": reduce_umap}
METHOD_ORDER = ["none", "pca", "autoencoder", "tsne", "umap"]
