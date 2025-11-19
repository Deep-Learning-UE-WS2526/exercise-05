import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("C:/Users/maris/Documents/Uni/IV/Deep Learning/exercise-05/data/train.csv")
df_rows, df_columns = df.shape


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
#print(df)
df = df.drop(columns="PassengerId")
df = df.drop(columns="Name")
print(df)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
def toNumeric(column):
    checklist = {}
    newColumn = []
    i = 0
    for element in column:
        if element not in checklist:
            checklist[element] = i
            i = i+1
        newColumn.append(checklist[element])
        #list comprehension?
    return newColumn

# for x in range(len(df.columns)):
#     for element in df[column]:
#         if not isinstance(element, (int, float)):
#             column = toNumeric(column)

#Versuch oben unfertig, deswegen mit hardcoding:
df["Sex"] = toNumeric(df["Sex"])
df["Cabin"] = toNumeric(df["Cabin"])
df["Ticket"] = toNumeric(df["Ticket"])
df["Embarked"] = toNumeric(df["Embarked"])
print(df)

# 3. Remove all rows that contain missing values
for row in df:
    for element in row:
        print(element)
        print(pd.isna(element))
        if pd.isna(element):
            df.drop(row)

#Leider keine Zeit mehr für die übrigen Aufgaben.

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.