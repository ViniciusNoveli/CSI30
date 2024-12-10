from sklearn.tree import DecisionTreeClassifier, plot_tree
import csv

def tree_sort():


    with open("treino_sinais_vitais_com_label.txt", "r") as training_file:
        labeled_data = [list(map(float, row)) for row in csv.reader(training_file, delimiter=",")]
        labeled_features = [row[3:-2] for row in labeled_data]
        labeled_target = [row[-1] for row in labeled_data]

    with open("treino_sinais_vitais_sem_label.txt", "r") as blind_testing_file:
        unlabeled_data = [list(map(float, row)) for row in csv.reader(blind_testing_file, delimiter=",")]
        data = [row for row in unlabeled_data]
        unlabeled_data = [row[3:-1] for row in data]
        print(unlabeled_data)

    clf = DecisionTreeClassifier(criterion='gini')

    clf.fit(labeled_features, labeled_target)

    predicted_target = clf.predict(unlabeled_data)

    print(predicted_target[:300])

    with open("resultadosID3.txt", "w") as f:
        for value in predicted_target:
            value = int(value)
            f.write(str(value) + "\n")


def main():
    tree_sort()

if __name__ == '__main__':
    main()
