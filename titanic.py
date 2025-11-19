import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_and_prepare(path: str) -> pd.DataFrame:
	"""Load Titanic data and perform preprocessing (Step 3).

	Steps:
	- Remove irrelevant columns: Name, PassengerId
	- Encode non-numeric columns (Sex, Cabin, Ticket, Embarked) into numeric form
	  using pandas factorize (simple label encoding). Missing categories become -1.
	- Drop rows with any remaining missing values.
	"""
	df = pd.read_csv(path)

	# 1. Remove columns
	df = df.drop(columns=["Name", "PassengerId"])  # raises if missing -> intentional

	# 2. Encode non-numeric columns
	to_encode = ["Sex", "Cabin", "Ticket", "Embarked"]
	for col in to_encode:
		# factorize returns (labels, uniques). Fill NaN beforehand so factorize assigns code.
		df[col] = pd.factorize(df[col].fillna("__MISSING__"))[0]

	# 3. Remove rows with missing numeric values (e.g. Age, Fare may have NaNs)
	df = df.dropna(axis=0, how="any")
	return df


def split_and_train(df: pd.DataFrame):
	"""Perform Step 4: split features/labels, train/test, fit logistic regression, report metrics."""
	# 1. Separate features and target label
	y = df["Survived"]
	X = df.drop(columns=["Survived"])  # all remaining columns are features

	# 2. Train/test split
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	# 3. Initialize and fit logistic regression
	model = LogisticRegression(solver="liblinear", random_state=42)
	model.fit(X_train, y_train)

	# 4. Evaluate
	y_pred = model.predict(X_test)
	report = classification_report(y_test, y_pred, digits=4)
	return model, report


def main():
	df_prepared = load_and_prepare("data/train.csv")
	model, report = split_and_train(df_prepared)
	print("Classification Report (Logistic Regression):")
	print(report)


if __name__ == "__main__":
	main()

