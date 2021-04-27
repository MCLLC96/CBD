import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import category_encoders as ce
import matplotlib.pyplot as plt
import graphviz


datos = pandas.read_csv("datasets\car_evaluation.csv")
col_names = ['buying', 'maint', 'doors',
             'persons', 'lug_boot', 'safety', 'class']
datos.columns = col_names
X = datos.drop(['class'], axis=1)
y = datos['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
encoder = ce.OrdinalEncoder(
    cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
clf_gini = DecisionTreeClassifier(
    criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_gini.fit(X_train, y_train))
dot_data = tree.export_graphviz(clf_gini, out_file="DecisionTree.gv",
                                feature_names=X_train.columns,
                                class_names=y_train,
                                filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data, filename="DecisionTree.gv", format="png")
graph.view()
