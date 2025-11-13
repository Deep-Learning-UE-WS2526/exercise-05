import pandas as pd
import numpy as np

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).

df.drop(columns=["Name", "PassengerId"], inplace=True)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".

df["Sex"] = np.where(df["Sex"] == "male", 0, 1) #male is 0, female 1
df["Cabin"] = np.where(df["Cabin"].isnull(), 0, 1) # no cabin info is 0, cabin info is 1
df["Ticket"] = df["Ticket"].astype('category').cat.codes
df["Embarked"] = df["Embarked"].astype('category').cat.codes

# 3. Remove all rows that contain missing values

df.dropna(inplace=True)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.

X = df.drop(columns=["Survived"])
y = df["Survived"]

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

from sklearn.metrics import precision_score, recall_score, f1_score
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")