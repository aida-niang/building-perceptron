# Building Perceptron
Projet réalisé dans le cadre de La Plateforme : coder un Perceptron en Python orienté objet, puis
l'appliquer à la détection du cancer du sein (dataset Breast Cancer Wisconsin).
## Contexte
Le Perceptron, inventé par Frank Rosenblatt en 1957, est le premier neurone artificiel capable
d'apprendre à partir d'exemples. Il calcule une somme pondérée de ses entrées, $z = w \cdot x + b$,
puis applique une fonction échelon : il prédit 1 si $z \geq 0$, sinon 0. Quand il se trompe, il
corrige ses poids avec la règle $w \leftarrow w + \eta \, (y - \hat{y}) \, x$.
Objectifs du projet :
1. Comprendre les notions de Machine Learning, de Deep Learning et de Perceptron (veille).
2. Coder ma propre classe `Perceptron` et la tester sur des données factices.
3. L'appliquer à un vrai problème médical et évaluer si un Perceptron suffit.
## Contenu du dépôt
| Fichier | Rôle |
|---|---|
| `perceptron.py` | Ma classe `Perceptron` (méthodes `fit`, `net_input`, `predict`) |
| `notebook.ipynb` | Le notebook complet : tests, nettoyage, analyse exploratoire, ACP,
modélisation, évaluation |
| `bcw_data.csv` | Le dataset Breast Cancer Wisconsin (Diagnostic) |
| `requirements.txt` | Les bibliothèques à installer |
## Lancer le projet
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Le fichier `bcw_data.csv` vient de Kaggle (« Breast Cancer Wisconsin (Diagnostic) Data Set ») et
est inclus dans le dépôt.
## Les données
- 569 patientes, 30 variables numériques calculées à partir d'images de cellules prélevées à
l'aiguille fine.
- 10 caractéristiques des noyaux (rayon, texture, périmètre, aire, lissage, compacité, concavité,
points concaves, symétrie, dimension fractale), chacune en moyenne (`_mean`), erreur standard (`_se`)
et pire valeur (`_worst`).
- Cible : `diagnosis`, M = malin (212 cas, 37 %) et B = bénin (357 cas, 63 %).
**Nettoyage :** suppression de `id` (inutile pour prédire) et de `Unnamed: 32` (vide). Aucune valeur
manquante, aucun doublon. Cible encodée en malin = 1, bénin = 0.
## Analyse exploratoire

- Les échelles sont très différentes (`area_mean` autour de 655, `smoothness_mean` autour de 0,1) :
standardisation obligatoire.
- La taille et la concavité des noyaux séparent bien les deux classes.
- 21 paires de variables sont corrélées à plus de 0,9 (par exemple rayon, périmètre et aire) :
beaucoup de redondance.
- Les valeurs extrêmes ont été gardées, car elles correspondent souvent à de vraies grosses tumeurs.
## Méthode
1. Séparation train / test 80 % / 20 % stratifiée (455 / 114 patientes).
2. Standardisation avec `StandardScaler`, ajustée sur le train seulement.
3. Réduction de dimension par ACP : 10 composantes gardent 95,2 % de la variance.
4. Entraînement de mon Perceptron (learning rate 0,01, 50 époques).
5. Réglage des hyperparamètres par validation croisée à 5 plis sur le train : tous les réglages
donnent entre 95,4 % et 96,9 %.
## Résultats sur le jeu de test
| Modèle | Accuracy | Recall malin |
|---|---|---|
| Mon Perceptron + ACP | 95,6 % | 97,6 % |
| Mon Perceptron sur les 30 variables | 93,9 % | 88,1 % |
| Perceptron de scikit-learn + ACP | 96,5 % | 95,2 % |
Avec l'ACP, mon Perceptron ne rate qu'un cancer sur 42 (1 faux négatif) et fait 4 fausses alertes sur
72 tumeurs bénignes (spécificité 94,4 %). L'AUC calculée à partir du score $z$ vaut 0,991.
## Conclusion
Le Perceptron fonctionne bien sur ce problème, parce que les données sont presque linéairement
séparables, à condition de standardiser les données et de réduire la dimension. Il reste limité pour
un vrai usage médical : il ne donne pas de probabilité, ne converge pas quand les données ne sont pas
parfaitement séparables et ne trouve pas forcément la meilleure frontière.
Pistes d'amélioration (non implémentées) : régression logistique (probabilités et seuil réglable),
SVM (marge maximale), perceptron multicouche (frontières non linéaires), forêts aléatoires,
pondération des classes pour réduire encore les faux négatifs.
## Outils utilisés
Python 3, NumPy, pandas, Matplotlib, Seaborn, scikit-learn (séparation, standardisation, ACP,
métriques et comparaison), Jupyter Notebook, Git et GitHub.
## Bibliographie
- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and
Organization in the Brain*. Psychological Review, 65(6).
- Novikoff, A. (1962). *On Convergence Proofs on Perceptrons*. Symposium on the Mathematical Theory
of Automata.
- Minsky, M. et Papert, S. (1969). *Perceptrons*. MIT Press.
- Street, W. N., Wolberg, W. H. et Mangasarian, O. L. (1993). *Nuclear feature extraction for breast
tumor diagnosis*. IS&T/SPIE.
- UCI Machine Learning Repository : Breast Cancer Wisconsin (Diagnostic) Data Set.
- Kaggle : Breast Cancer Wisconsin (Diagnostic) Data Set.
- Documentation scikit-learn : `StandardScaler`, `PCA`, `StratifiedKFold`, métriques de
classification.
- Vidéo « Perceptron Algorithm with Code Example - ML for beginners! » (lien donné dans le sujet).
- Vidéo « Gradient Descent - Simply Explained! ML for beginners with Code Example! » (lien donné dans
le sujet).
- Boruta_py : https://github.com/scikit-learn-contrib/boruta_py
