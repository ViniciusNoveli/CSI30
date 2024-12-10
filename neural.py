import pandas as pd
from numpy import ceil
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import csv

def neural_regressor():

    # Carrega os dados com label
    with open("treino_sinais_vitais_com_label.txt", "r") as training_file:
        labeled_data = [list(map(float, row)) for row in csv.reader(training_file, delimiter=",")]
        labeled_features = [row[3:-2] for row in labeled_data]
        labeled_target = [row[-1] for row in labeled_data]

    # Carrega os dados sem label
    with open("treino_sinais_vitais_sem_label.txt", "r") as blind_testing_file:
        unlabeled_data = [list(map(float, row)) for row in csv.reader(blind_testing_file, delimiter=",")]
        data = [row for row in unlabeled_data]
        unlabeled_data = [row[3:-1] for row in data]
        print(unlabeled_data)

    # Padronizar os dados de treinamento
    scaler = StandardScaler()
    labeled_features = scaler.fit_transform(labeled_features)

    # Treinar o modelo de regressão neural
    regressor = MLPRegressor(hidden_layer_sizes=(32, 64, 128, 64, 32, 8), activation='relu', max_iter=1500,
                            learning_rate='adaptive', learning_rate_init=0.001)
    regressor.fit(labeled_features, labeled_target)

    # Padronizar os dados de teste
    unlabeled_data = scaler.transform(unlabeled_data)

    # Aplicar a regressão nos dados de teste
    y_pred = regressor.predict(unlabeled_data)

    y_pred = [f'{value:.4f  }'.replace('.', ',') for value in y_pred]
    # Salvar valores previstos em um arquivo .txt
    with open("resultadosNeural", 'w') as file:
        file.write('\n'.join(y_pred))

def main():
    neural_regressor()

if __name__ == '__main__':
    main()