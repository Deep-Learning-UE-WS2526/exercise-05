import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(columns=["Name", "PassengerId"])

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
categorical_cols = ["Sex", "Cabin", "Ticket", "Embarked"]
df = pd.get_dummies(df, columns=categorical_cols)

# 3. Remove all rows that contain missing values
df = df.dropna()


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
X = df.drop("Survived", axis=1)   
y = df["Survived"]                

# 2. Secondly, we need to split training and test data. This can be done with the function
#    `sklearn.model_selection.train_test_split()` from the `scikit-learn` library.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver,
#    and fit it to the training data.
model = LogisticRegression(solver="liblinear", max_iter=1000)
model.fit(X_train, y_train)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)
