import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
del df["Name"]
del df["PassengerId"]

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are 
# "Sex", 
df["Sex"] = df["Sex"].replace(["female","male"],[0,1])
# "Cabin",
df.dropna(inplace = True) # löscht alle missing values, funktioniert sonst nicht
for x in df.index:
    newCabin = ""
    for c in df.loc[x, "Cabin"]:
        if c.isalpha():
            newCabin = newCabin + str(ord(c))
            newCabin = newCabin + "."
        elif c.isdigit():
            newCabin = newCabin + str(c)
        else:
            break
    df.loc[x, "Cabin"] = newCabin
df["Cabin"] = pd.to_numeric(df["Cabin"], errors="coerce")
# "Ticket",
df["Ticket"] = pd.to_numeric(df["Ticket"], errors="coerce")
# and "Embarked".
df["Embarked"] = df["Embarked"].replace(["S","C", "Q"],[0,1,2])

# 3. Remove all rows that contain missing values
df.dropna(inplace = True)
print(df)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels.
# This can be done easily with `pandas`.

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

