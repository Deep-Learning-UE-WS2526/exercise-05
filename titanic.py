import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("exercise-05/data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(['Name', 'PassengerId'], axis=1)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and 
# "Embarked".


df = pd.get_dummies(df, columns=['Sex', 'Cabin', 'Ticket', 'Embarked'], dummy_na=False)

# 3. Remove all rows that contain missing values
df = df.dropna()
print(df)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. T
# his can be done easily with `pandas`.

# Unabhängige Variablen zum Training
X = df.drop('Survived', axis=1)

#Abhängige Variable, die es herauszufinden gilt
y = df['Survived']


# 2. Secondly, we need to split training and test data. This can be done with the function 
# [`sklearn.model_selection.train_test_split()`]
# (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) 
# from the `scikit-learn` library.

# Features und Labels werden voneinander getrennt (X,Y), dann jeweils in ein Trainings- und Testset gepackt
# Man kann nicht einfahc eine ganze Tabelle geben und sagen lern das und sag mir wie man auf Survived kommt, sondern
# Lern anhand dieser Features (x), wie man auf das Label hier (y) kommt&
# Sage vorraus anhand dieser Features (x), was bei herauskommt (y)
from sklearn.model_selection import train_test_split
# Hier sowohl Train und Test, "getrennt", damit man die jeweiligen sets getrennt voneinander testen und trainieren kann
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
from sklearn.linear_model import LogisticRegression

# Unter clf erstellt man ein LogisticRegression Model mit liblinear solver und das ganze fittet man auf x_train, y_train
# Der random_state bestimmt, wie reproduzierbar die Zufallsprozesse sind.
# Er setzt den Startpunkt für den Zufallsalgorithmus, sodass Train/Test-Aufteilungen immer gleich bleiben.
clf = LogisticRegression(random_state=0, solver = 'liblinear').fit(X_train, y_train)



# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
from sklearn.metrics import precision_recall_fscore_support

y_pred = clf.predict(X_test)    # die Vorhersage von Werten mithilfe des Modells mit x_test als Eingabe
y_true = y_test                 # die y trues sind alle richtigen Werte aus dem y_test Set


# weighted, weil man ja beide Klassen behandeln will (survided, nicht survived gleichermaßen betrachtet)
# Ansonsten binary, wenn man nur einen bestimmten Wert herausfinden will (nur survived)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

