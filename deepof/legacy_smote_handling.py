"""
Legacy imblearn compatibility + lightweight resampling utilities.

- Legacy unpickling: allows loading old pickles that reference
  imblearn.pipeline.Pipeline and imblearn.over_sampling.SMOTE without depending
  on imblearn/imbalanced-learn.

- Training-time resampling: SimpleSMOTE + ResampledClassifier (sklearn-native).
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# ---------------------------------------------------------------------
# Legacy imblearn unpickling shims (inference only)
# ---------------------------------------------------------------------

class _IdentityResampler(BaseEstimator):
    """Inference-time stand-in for samplers (e.g., SMOTE)."""
    def fit_resample(self, X, y):
        return X, y


def _pipeline_transform_skip_samplers(steps, X):
    """Apply transform of each non-sampler step (skip fit_resample-only steps)."""
    Xt = X
    for _, step in steps:
        if step is None or step == "passthrough":
            continue
        # sampler: has fit_resample but no transform
        if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
            continue
        Xt = step.transform(Xt)
    return Xt


class _LegacyImblearnPipeline(SklearnPipeline):
    """Stand-in for imblearn.pipeline.Pipeline, sufficient for inference."""

    def __setstate__(self, state):
        # state is a dict restored from the pickle; ensure key exists
        if isinstance(state, dict):
            state.setdefault("transform_input", None)
        super().__setstate__(state)
        
    def _call_final(self, method: str, X, **params):
        Xt = _pipeline_transform_skip_samplers(self.steps[:-1], X)
        final_est = self.steps[-1][1]
        return getattr(final_est, method)(Xt, **params)

    def predict(self, X, **params):
        return self._call_final("predict", X, **params)

    def predict_proba(self, X, **params):
        return self._call_final("predict_proba", X, **params)

    def decision_function(self, X, **params):
        return self._call_final("decision_function", X, **params)

    def score(self, X, y=None, **params):
        Xt = _pipeline_transform_skip_samplers(self.steps[:-1], X)
        return self.steps[-1][1].score(Xt, y, **params)


class _ImblearnCompatUnpickler(pickle.Unpickler):
    """Unpickler that rewires imblearn references to local stand-ins."""
    def find_class(self, module: str, name: str) -> Any:
        if module == "imblearn.pipeline" and name == "Pipeline":
            return _LegacyImblearnPipeline

        # SMOTE has varied internal module paths across imblearn versions.
        if name == "SMOTE" and module.startswith("imblearn.over_sampling"):
            return _IdentityResampler

        return super().find_class(module, name)


def load_pickle_compat(path: str):
    """Load an old pickle that may reference imblearn objects."""
    with open(path, "rb") as f:
        return _ImblearnCompatUnpickler(f).load()


# ---------------------------------------------------------------------
# Training-time resampling utilities (sklearn-native)
# ---------------------------------------------------------------------

class SimpleSMOTE(BaseEstimator):
    """Small, dependency-free SMOTE-like resampler (dense numeric arrays)."""

    def __init__(self, k_neighbors: int = 5, random_state: Optional[int] = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_out = [X]
        y_out = [y]

        for cls, n_cls in zip(classes, counts):
            n_to_gen = int(max_count - n_cls)
            if n_to_gen <= 0:
                continue

            Xc = X[y == cls]
            n_samples = Xc.shape[0]

            # Too few samples -> duplicate
            if n_samples < 2:
                idx = rng.randint(0, n_samples, size=n_to_gen)
                X_out.append(Xc[idx])
                y_out.append(np.full(n_to_gen, cls))
                continue

            k = min(self.k_neighbors, n_samples - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
            neigh_idx = nn.kneighbors(Xc, return_distance=False)[:, 1:]

            X_syn = np.empty((n_to_gen, X.shape[1]), dtype=X.dtype)
            for i in range(n_to_gen):
                a = rng.randint(0, n_samples)
                b = rng.choice(neigh_idx[a])
                lam = rng.rand()
                X_syn[i] = Xc[a] + lam * (Xc[b] - Xc[a])

            X_out.append(X_syn)
            y_out.append(np.full(n_to_gen, cls))

        return np.vstack(X_out), np.concatenate(y_out)


class ResampledClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier; resample (X, y) inside fit before training.

    Works inside sklearn.pipeline.Pipeline and with sklearn.model_selection tools.
    """

    def __init__(self, estimator, resampler: Optional[Any] = None):
        self.estimator = estimator
        self.resampler = resampler

    def fit(self, X, y, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.estimator_ = clone(self.estimator)

        if self.resampler is None:
            Xr, yr = X, y
        else:
            self.resampler_ = clone(self.resampler)
            Xr, yr = self.resampler_.fit_resample(X, y)

        self.estimator_.fit(Xr, yr, **fit_params)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(yr))
        return self

    def __getattr__(self, name: str):
        """
        Delegate unknown attributes/methods to the fitted estimator.

        This removes the need to re-define predict/predict_proba/decision_function/score
        unless you want custom input validation.
        """
        if name.endswith("_"):
            raise AttributeError(name)
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, name)

    # Optional: keep predict with validation if you prefer strict checks.
    def predict(self, X):
        check_is_fitted(self, "estimator_")
        X = check_array(X, accept_sparse=False)
        return self.estimator_.predict(X)