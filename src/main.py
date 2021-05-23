from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import category_encoders as ce
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# Función para importar, analizar y normalizar los datos
def import_data(file, column_names):

    data = pd.read_csv(file, header = None, names = column_names)

    data.dropna(axis=0)

    print('\n DATASET PREVIEW:\n', data.head(), '\n')

    print ('SHAPE OF DATASET (ROWS, COLUMNS):\n', data.shape, '\n')

    print('VALUES FRECUENCY BY COLUMN:')

    for col in data.columns:
        print(data[col].value_counts(), '\n')
    
    return data


# Función para codificar las variables categóricas y separar el conjunto de datos en entrenamiento y prueba.
def split_encode_data(data):

    X = data.drop(['class'], axis=1)

    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20)

    encode = ce.OrdinalEncoder()

    X_train = encode.fit_transform(X_train)

    X_test = encode.transform(X_test)
    	
    return X_train, X_test, y_train, y_test, encode

# Función de entrenamiento del árbol de decisión a partir de los datos de entrenamiento 
def decision_tree(X_train, y_train):

    decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 40, class_weight='balanced')

    model = decision_tree.fit(X_train, y_train)

    feature_scores = pd.Series(model.feature_importances_, index = X_train.columns).sort_values(ascending = False)

    print('FEATURE SCORES:')
    
    print(feature_scores, '\n')

    return model

# Función que a partir de un árbol de decisión y datos de entrenamiento devuelve la predicción
def predict(model, X_test):

    y_pred = model.predict(X_test)

    return y_pred

# Función para las métricas
def accuracy(y_test, y_pred):

    print('CONFUSION MATRIX: \n', confusion_matrix(y_test, y_pred), '\n')
      
    print ('ACCURACY :\n ', accuracy_score(y_test, y_pred) * 100, '\n')
      
    print('REPORT :\n', classification_report(y_test, y_pred))

# Función para predecir de forma manual
def manual_test(model, encode):

    print('PRUEBA MANUAL: ')

    aux = [
        'Precio de compra (low, high, med, vhigh): ', 
        'Precio de mantenimiento (low, high, med, vhigh): ', 
        'Número de puertas (2, 3, 4, 5more): ', 
        'Número de personas (2, 4, more): ', 
        'Tamaño del maletero (small, med, big): ', 
        'Nivel de seguridad estimado (low, high, med): ']
    

    aux2 = ['low, high, med, vhigh', 'low, high, med, vhigh', '2, 3, 4, 5more', '2, 4, more', 'small, med, big', 'low, high, med']

    enc = encode.get_params('mapping')['mapping']

    test = []

    for i in range(0, len(aux)):

        while(True):

            options = aux2[i].split(sep = ', ')
            
            value = str(input(aux[i])).lstrip().lower()
    
            if(value in options):

                test.append(enc[i]['mapping'][value])

                break

            else:

                print('El valor introducido es incorrecto.')


    pred = model.predict([test])

    print('\nPredicción del nivel de seguridad: ', pred[0], '\n')

# Función main
def main():
      
    data = import_data('car_evaluation.csv', ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    
    X_train, X_test, y_train, y_test, encode = split_encode_data(data)
    
    model = decision_tree(X_train, y_train)

    y_pred = predict(model, X_test)

    accuracy(y_test, y_pred)

    manual_test(model, encode)


if __name__=="__main__":
    main()