import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import csv

with open("treino_sinais_vitais_com_label.txt", "r") as training_file:
    labeled_data = [list(map(float, row)) for row in csv.reader(training_file, delimiter=",")]
    labeled_features = [row[3:-2] for row in labeled_data]
    labeled_target = [row[-1] for row in labeled_data]

with open("treino_sinais_vitais_sem_label.txt", "r") as blind_testing_file:
    unlabeled_data = [list(map(float, row)) for row in csv.reader(blind_testing_file, delimiter=",")]
    data = [row for row in unlabeled_data]
    unlabeled_data = [row[3:-1] for row in data]

# Separar características (X) e rótulos (y) no conjunto rotulado

# Dividir o conjunto rotulado em treino e teste para validação
X_train, X_test, y_train, y_test = train_test_split(labeled_features, labeled_target, test_size=0.2, random_state=42)

# Criar e treinar o modelo Random Forest
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Avaliar o modelo no conjunto de teste
y_pred_test = modelo.predict(X_test)
print("Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred_test))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred_test))

# Fazer previsões para o conjunto não rotulado
previsoes_nao_rotulados = modelo.predict(unlabeled_data)

# Adicionar os rótulos previstos ao conjunto não rotulado
for i in range(len(unlabeled_data)):
    unlabeled_data[i].append(int(previsoes_nao_rotulados[i]))

# Salvar o conjunto não rotulado com as previsões
with open("resultadosRF.txt", "w") as f:
    for line in unlabeled_data:
        f.write(str(line) + "\n")

