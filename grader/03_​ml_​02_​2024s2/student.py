import warnings  # DO NOT modify this line

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning  # DO NOT modify this line
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings(
    "ignore", category=ConvergenceWarning
)  # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path):  # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=",")
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self):  # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        # TODO: Paste your code here
        return self.df.shape[0]

    def Q2(self):  # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        # TODO: Paste your code here
        num_cols = self.df.select_dtypes(include=["number"])
        obj_cols = self.df.select_dtypes(include=["object"])

        return (num_cols.shape[1], obj_cols.shape[1])

    def Q3(self):  # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        # TODO: Paste your code here
        no, yes = self.df["y"].value_counts()
        return (round(no / self.df.shape[0], 3), round(yes / self.df.shape[0], 3))

    def Q4(self):  # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        # TODO: Paste your code here
        self.df.drop_duplicates(inplace=True)

        return self.df.shape

    def Q5(self):  # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values.
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        # TODO: Paste your code here
        self.df.drop_duplicates(inplace=True)

        flat_cols = self.df.apply(
            lambda col: col.value_counts(normalize=True).max() >= 0.90
        )
        self.df.drop(columns=flat_cols[flat_cols].index, inplace=True)

        y = self.df.pop("y")
        X = self.df

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)

        return self.X_train.shape, self.X_test.shape

    def onehot_cols(self, X: pd.DataFrame, nominal_cols: pd.Series):
        enc = OneHotEncoder(handle_unknown="ignore")
        enc_df = pd.DataFrame(enc.fit_transform(X[nominal_cols]).toarray())

        unique_vals = enc.categories_
        new_col_names = []
        cou = 0
        for i, vals in enumerate(unique_vals):
            cou += len(vals)
            for j, val in enumerate(vals):
                new_col_names.append(f"{nominal_cols[i]}_{val}")

        enc_df.columns = new_col_names
        X = pd.concat([X, enc_df], axis=1)
        X.drop(columns=nominal_cols, inplace=True)

        return X

    def Q6(self):
        """
        Problem 6:
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the nominal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7}
                Hint: Use One hot encoder or pd.dummy to encode nominal category
            return the shape of X_train.
        """
        # TODO: Paste your code here
        self.Q5()

        # impute
        num_impute = SimpleImputer(missing_values=np.nan, strategy="mean")
        num_cols = self.df.select_dtypes(include=["number"]).columns
        self.X_train[num_cols] = pd.DataFrame(
            num_impute.fit_transform(self.X_train[num_cols])
        )
        self.X_test[num_cols] = pd.DataFrame(
            num_impute.transform(self.X_test[num_cols])
        )

        cat_impute = SimpleImputer(missing_values="unknown", strategy="most_frequent")
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        self.X_train[cat_cols] = pd.DataFrame(
            cat_impute.fit_transform(self.X_train[cat_cols])
        )
        self.X_test[cat_cols] = pd.DataFrame(
            cat_impute.transform(self.X_test[cat_cols])
        )

        poutcome_imput = SimpleImputer(
            missing_values="nonexistent", strategy="most_frequent"
        )
        self.X_train["poutcome"] = pd.DataFrame(
            poutcome_imput.fit_transform(self.X_train[["poutcome"]])
        )
        self.X_test["poutcome"] = pd.DataFrame(
            poutcome_imput.transform(self.X_test[["poutcome"]])
        )

        # education col nominal -> numeric
        education_order = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "professional.course": 6,
            "university.degree": 7,
        }

        self.X_train["education"] = self.X_train["education"].map(education_order)
        self.X_test["education"] = self.X_test["education"].map(education_order)

        nominal_cols = pd.Series([c for c in cat_cols if c != "education"])
        self.X_train = self.onehot_cols(self.X_train, nominal_cols)
        self.X_test = self.onehot_cols(self.X_test, nominal_cols)

        self.y_train = self.y_train.map({"yes": 1, "no": 0})
        self.y_test = self.y_test.map({"yes": 1, "no": 0})

        return self.X_train.shape

    def Q7(self):
        """Problem7: Use Logistic Regression as the model with
        random_state=2025,
        class_weight='balanced' and
        max_iter=500.
        Train the model using all the remaining available variables.
        What is the macro F1 score of the model on the test data? in 2 digits
        """
        # TODO: Paste your code here
        self.Q6()

        logmodel = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=2025
        )
        logmodel.fit(self.X_train, self.y_train)

        predictions = logmodel.predict(self.X_test)
        report = classification_report(
            self.y_test, predictions, output_dict=True, digits=2
        )

        return round(report["macro avg"]["f1-score"], 2)
