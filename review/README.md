# Visualization

# Data Engineering

# Data Science
```py
vdo_df.drop_duplicates(inplace=True)
# count only unique titles
vdo_df[vdo_df["dislikes"] > vdo_df["likes"]]["title"].nunique()

```