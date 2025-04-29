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



```

#