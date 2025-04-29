# Visualization

# Data Engineering

# Data Science
## Pandas
```py
df.drop_duplicates(inplace=True)
# count only unique titles
df[df["dislikes"] > df["likes"]]["title"].nunique()

# many conditions (don't forget to use parentheses)
df[ (df["trending_date"] == "18.22.01") & (df["comment_count"] > 10000) ]

# group by trending date, get the date with lowest mean comment count
df = vdo_df.groupby("trending_date")
df["comment_count"].mean().idxmin()

# create df from json
with open("/data/GB_category_id.json") as fd:
        cat = json.load(fd)
cat_list = []
for item in cat["items"]:
    cat_list.append((int(item["id"]), item["snippet"]["title"]))
cat_df = pd.DataFrame(cat_list, columns=["id", "category"])
# merge dataframes
vdo_df_with_cat = vdo_df.merge(cat_df, left_on="category_id", right_on="id")
# group by (trending_date, category), each row is the sum of views
grouped_df = (
    vdo_df_with_cat.groupby(["trending_date", "category"])["views"]
    .sum()
    .reset_index() # to make it a dataframe
)
# count no. of days sports has more views than comedy
sport_df = grouped_df[grouped_df["category"] == "Sports"].reset_index()
comedy_df = grouped_df[grouped_df["category"] == "Comedy"].reset_index()
return (sport_df['views'] > comedy_df["views"]).sum()

# remove columns with more than 50% missing values
half_missing = df.isnull().sum() > df.shape[0] // 2
df.drop(df.columns[half_missing], axis=1, inplace=True)

# remove columns with more than 70% of the same value
flat_cols = df.select_dtypes(include="object").apply(
    lambda col: col.value_counts(normalize=True).max() > 0.7
)
df.drop(columns=flat_cols[flat_cols].index, inplace=True)

# remove rows with missing values in specific columns
df.dropna(subset=["Survived"], inplace=True)

# get quartiles
q1, q3 = np.percentile(df["Fare"], [25, 75])
# replace outliers with the min/max of the quartiles
df.loc[df["Fare"] < min_outlier, "Fare"] = min_outlier

# get the number of missing values in each column
null_counts = df.isnull().sum()

# impute missing values with the mean
num_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
df[["Pclass", "Age", "SibSp"]] = pd.DataFrame(
    num_imp.fit_transform(df[["Pclass", "Age", "SibSp"]])
)

# one-hot encode categorical variables
enc = OneHotEncoder(handle_unknown="ignore")
nominal_columns = ["Embarked"]
enc_df = pd.DataFrame(enc.fit_transform(df[nominal_columns]).toarray())
unique_vals = enc.categories_[0]
new_col_names = [f"Embarked_{val}" for val in unique_vals]
enc_df.columns = new_col_names
df = pd.concat([df, enc_df], axis=1)

# apply lambda function to a column
df["Survived"] = df["Survived"].apply(lambda x: 1.0 if x > 0.5 else 0.0)
# split data into train and test sets
y = df.pop("Survived")
X = df
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=123
)
survived_train = y_train.sum() / y_train.shape[0]
```

## Random Forest
```py
# clean
hw.dropna(subset=['label'], inplace=True)
hw.drop(columns=['id','gill-attachment', 'gill-spacing'], inplace=True)
hw.reset_index(inplace=True)


```
