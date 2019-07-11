from __future__ import division
from sklearn.ensemble import RandomForestClassifier
from time import time
import pandas as pd
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import os
from sklearn.tree import export_graphviz
import six
import pydot
from sklearn import tree



class Classifier():

    def __init__(self):
        self.model = RandomForestClassifier(criterion="entropy", n_estimators=101, random_state=47,
                                            max_depth=3, max_features=4)

    def fit(self, X, y, graph=False):
        print('Fitting model', end='...')
        start = time()
        self.model.fit(X, y)
        end = time()
        print('done')
        print('Training time took:', (end - start), 'sec')

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def plot_tree(self, data, y, i, num):
        cols = select_best_var(data, y, num, plot_=False)
        #dotfile = six.StringIO()
        estimator = self.model.estimators_[i]
        export_graphviz(estimator,
                        out_file='tree.dot',
                        filled=True,
                        rounded=True,
                        feature_names=cols,
                        class_names=['0', '1'])

        (graph,) = pydot.graph_from_dot_file('tree.dot')
        graph.write_png('tree.png')
        os.system('dot -Tpng tree.dot -o tree.png')
        return Image(filename='tree.png')


def select_best_var(data, y, num, plot_=True):
    importance = pd.DataFrame()
    importance['name'] = data.columns

    model = RandomForestClassifier(n_estimators=100)

    model.fit(data, y)
    importance['importance'] = model.feature_importances_

    if plot_:
        (pd.Series(model.feature_importances_, index=data.columns).nlargest(num).plot(kind='barh',
                                                                                      title='Feature importance of top features'))

    best_var = importance.sort_values(by=['importance'], ascending=False, inplace=False)
    best_var = best_var['name'][:num]
    best_var = best_var.values

    return best_var