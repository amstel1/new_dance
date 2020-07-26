from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
plt.rcParams["figure.figsize"] = 7,7
plt.style.use('ggplot')
import numpy as np

from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 8)

lr = LogisticRegression(n_jobs=-1)

rf = RandomForestClassifier(n_jobs=-1)
ada = AdaBoostClassifier()
lgbm = LGBMClassifier()
xgb = XGBClassifier(n_jobs=-1)
cat = CatBoostClassifier(verbose = False)
etc=ExtraTreesClassifier()
gbc =GradientBoostingClassifier()
nb = GaussianNB()
mnb=MultinomialNB()
cnb1=ComplementNB()
bnb=BernoulliNB()
cnb2=CategoricalNB()
qda = QuadraticDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis()
rccv = RidgeClassifierCV()
rc = RidgeClassifier()
pf = PolynomialFeatures(interaction_only=False, degree=1, include_bias=False,)
sc = StandardScaler()

classifiers = [rc, rccv,lda, cat, lgbm, xgb, ada, rf, lr, etc, gbc]
for clf in classifiers:
    #pipe = Pipeline([('impute',si),('extract features',pf), ('scale', sc), ('classify', clf)]) #, ('extract features',pf), ('scale', sc)
    pipe = Pipeline([('classify',clf)])#, ('std',sc)
    cvs = cross_val_score(pipe, X[['a','b','c']], y, scoring='roc_auc', cv=3, n_jobs=-1)
    print(np.std(cvs), np.mean(cvs), clf)
    print()