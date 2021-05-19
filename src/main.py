from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
X = datos.drop(['safety'], axis=1)
y = datos['safety']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
encoder = ce.OrdinalEncoder(
    cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'class'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
clf_en = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_en.fit(X_train, y_train))


# -----------------Predict--------------------

valores = [('vhigh', 'vhigh', 2, 2, 'small', 'unacc')]
columnas = ['buying', 'maint', 'doors',
            'persons', 'lug_boot', 'class']
dato = pandas.DataFrame(valores, columns=columnas)
trans = encoder.transform(dato)
res = clf_en.predict(trans)
print('El resultado es:')
print(res)
print("")

# ------------------Quality---------------------

cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix:\n', cm)


print('Model accuracy score with criterion entropy: {0:0.4f}\n'. format(
    accuracy_score(y_test, y_pred_en)))

y_pred_train_en = clf_en.predict(X_train)
y_pred_train_en
print('Training-set accuracy score: {0:0.4f}\n'. format(
    accuracy_score(y_train, y_pred_train_en)))

print('Training set score: {:.4f}\n'.format(clf_en.score(X_train, y_train)))
print('Test set score: {:.4f}\n'.format(clf_en.score(X_test, y_test)))

print(classification_report(y_test, y_pred_en))
