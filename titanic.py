import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 3.1 Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(["Name", "PassengerId"], axis="columns")

# 3.2 Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df["Sex"] = df["Sex"].replace(["male", "female"], [0, 1])
df["Cabin"] = df["Cabin"].str.extract("(\d+)")
df["Ticket"] = df["Ticket"].str.extract("(\d+)")
df["Embarked"] = df["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])

'''  [sex]  male: 0 | female: 1 
[embarked]  S: 0 | C: 1 | K: 2 '''

# 3.3 Remove all rows that contain missing values
df = df.dropna(axis="rows")

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with 'pandas'.
x = df.drop("Survived", axis="columns")
y = df["Survived"]

# 2. Secondly, we need to split training and test data. This can be done with the function ['sklearn.model_selection.train_test_split()'](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the 'scikit-learn' library.
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a 'liblinear' solver, and fit it to the training data.
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=16)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from 'scikit-learn'.
from sklearn.metrics import precision_score, recall_score, f1_score
print("precision = ", precision_score(y_test, y_pred, average="weighted"))
print("recall = ", recall_score(y_test, y_pred, average="weighted"))
print("f-score = ", f1_score(y_test, y_pred, average="weighted"))

