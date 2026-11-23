import numbers

import numpy as np


class Perceptron:
    """Classifieur binaire linéaire entraîné avec la règle du Perceptron."""

    def __init__(
        self,
        learning_rate=0.01,
        n_epochs=100,
        shuffle=True,
        random_state=42
    ):
        if (
            isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(learning_rate, numbers.Real)
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate doit être strictement positif.")
        if (
            isinstance(n_epochs, (bool, np.bool_))
            or not isinstance(n_epochs, numbers.Integral)
            or n_epochs < 1
        ):
            raise ValueError(
                "n_epochs doit être un entier supérieur ou égal à 1."
            )

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state

        self.weights_ = None
        self.bias_ = 0.0
        self.errors_ = []
        self.n_epochs_run_ = 0
        self.converged_ = False

    # --------------------------------------------------
    # Méthodes internes
    # --------------------------------------------------

    @staticmethod
    def _activation(z):
        """Fonction échelon : 1 si z >= 0, sinon 0."""
        return np.where(z >= 0.0, 1, 0)

    @staticmethod
    def _check_X(X):
        """Vérifie la forme des données."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X doit être un tableau à 2 dimensions "
                "(n_samples, n_features)."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X doit contenir uniquement des valeurs finies.")

        return X

    @staticmethod
    def _check_y(y):
        """Vérifie que les étiquettes sont binaires."""
        y = np.asarray(y).ravel()

        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(
                "y doit contenir uniquement les valeurs 0 et 1."
            )

        return y

    def _check_fitted(self):
        """Vérifie que le modèle a été entraîné."""
        if self.weights_ is None:
            raise RuntimeError(
                "Le modèle n'est pas entraîné : "
                "appelez d'abord fit(X, y)."
            )

    # --------------------------------------------------
    # Méthodes publiques
    # --------------------------------------------------

    def net_input(self, X):
        """Calcule z = w.x + b."""
        self._check_fitted()
        X = self._check_X(X)

        if X.shape[1] != len(self.weights_):
            raise ValueError(
                "X doit contenir le même nombre de caractéristiques "
                "que les données d'entraînement."
            )

        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        """Prédit la classe 0 ou 1."""
        return self._activation(self.net_input(X))

    def fit(self, X, y):
        """Entraîne le Perceptron."""

        X = self._check_X(X)
        y = self._check_y(y)

        if len(X) == 0:
            raise ValueError("X ne doit pas être vide.")

        if len(X) != len(y):
            raise ValueError(
                "X et y doivent contenir le même nombre d'exemples."
            )

        rng = np.random.default_rng(self.random_state)

        # Initialisation des paramètres
        self.weights_ = rng.normal(
            loc=0.0,
            scale=0.01,
            size=X.shape[1]
        )
        self.bias_ = 0.0

        self.errors_ = []
        self.n_epochs_run_ = 0
        self.converged_ = False

        # Entraînement
        for epoch in range(self.n_epochs):

            if self.shuffle:
                indices = rng.permutation(len(X))
            else:
                indices = np.arange(len(X))

            n_errors = 0

            for i in indices:
                z = np.dot(X[i], self.weights_) + self.bias_
                y_hat = self._activation(z)

                # Règle d'apprentissage du Perceptron
                update = self.learning_rate * (y[i] - y_hat)

                self.weights_ += update * X[i]
                self.bias_ += update

                if update != 0.0:
                    n_errors += 1

            self.errors_.append(n_errors)
            self.n_epochs_run_ = epoch + 1

            # Arrêt si aucune erreur pendant l'époque
            if n_errors == 0:
                self.converged_ = True
                break

        return self

    def score(self, X, y):
        """Calcule l'accuracy."""
        y = self._check_y(y)

        if len(y) == 0:
            raise ValueError("y ne doit pas être vide.")

        predictions = self.predict(X)

        if len(predictions) != len(y):
            raise ValueError(
                "X et y doivent contenir le même nombre d'exemples."
            )

        return float(np.mean(predictions == y))

    def __repr__(self):
        return (
            f"Perceptron(learning_rate={self.learning_rate}, "
            f"n_epochs={self.n_epochs}, "
            f"shuffle={self.shuffle}, "
            f"random_state={self.random_state})"
        )
