import numpy as np

class Perceptron:
    "Mon perceptron de Rosenblatt pour classer en 0 ou 1."

    def __init__(self, learning_rate=0.01, n_iter=50, random_state=42):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        "Entraine le perceptron sur X (les donnees) et y (les reponses 0 ou 1)."
        X = np.array(X)
        y = np.array(y)
        generateur = np.random.RandomState(self.random_state)
        nb_colonnes = X.shape[1]
        self.weights_ = generateur.normal(0, 0.01, nb_colonnes)
        self.bias_ = 0.0
        self.errors_ = []

        for epoque in range(self.n_iter):
            erreurs = 0
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]
                prediction = self.predict(xi)
                erreur = yi - prediction
                self.weights_ = self.weights_ + self.learning_rate * erreur * xi
                self.bias_ = self.bias_ + self.learning_rate * erreur
                if erreur != 0:
                    erreurs = erreurs + 1
            self.errors_.append(erreurs) 
            if erreurs == 0:
                break
        return self

    def net_input(self, X):
        """Calcule z = w . x + b"""
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        """Fonction echelon : 1 si z >= 0, sinon 0"""
        z = self.net_input(X)
        return np.where(z >= 0, 1, 0)
