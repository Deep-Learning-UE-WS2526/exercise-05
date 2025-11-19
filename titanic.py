from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_score, recall_score, f1_score
import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(['PassengerId', 'Name'])

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 3. Remove all rows that contain missing values
df = df.dropna

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.drop('Survived', axis=1)
y = df['Survived'] # y als Zielspalte für Kategorie 'Survived' und x für alle normalen Features

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, y_train, x_test, y_test = train_test_split(x,y, train_size=0.6, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver (Optimierungsalgorithmus in sklearn für LogisticRegression), and fit it to the training data.
model = LogisticRegression(solver='liblinear', max_iter=1000) # Objekt der Klasse erstellen
model.fit(x_train, y_train) # Training starten

y_pred = model.predict(x_test) # Vorhersagen bekommen nach Training mit fit()

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred)
