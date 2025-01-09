import json

import pandas as pd
from pandas import DataFrame

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
"""
vdo_df = pd.read_csv("/data/GBvideos.csv")
# vdo_df = pd.read_csv("USvideos.csv")

with open("/data/GB_category_id.json") as fd:
    # with open("US_category_id.json") as fd:
    cat = json.load(fd)


def Q1():
    """
    1. How many rows are there in the GBvideos.csv after removing duplications?
    - To access 'GBvideos.csv', use the path '/data/GBvideos.csv'.
    """
    # TODO: Paste your code here
    vdo_df.drop_duplicates(inplace=True)
    return vdo_df.shape[0]


def Q2(vdo_df):
    """
    2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    # TODO: Paste your code here
    vdo_df.drop_duplicates(inplace=True)
    return vdo_df[vdo_df["dislikes"] > vdo_df["likes"]]["title"].unique().shape[0]


def Q3(vdo_df):
    """
    3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    """
    # TODO: Paste your code here
    vdo_df.drop_duplicates(inplace=True)
    return vdo_df[
        (vdo_df["trending_date"] == "18.22.01") & (vdo_df["comment_count"] > 10000)
    ].shape[0]


def Q4(vdo_df: DataFrame):
    """
    4. Which trending date that has the minimum average number of comments per VDO?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    # TODO:  Paste your code here
    vdo_df.drop_duplicates(inplace=True)
    df = vdo_df.groupby("trending_date")
    return df["comment_count"].mean().idxmin()


def Q5(vdo_df):
    """
    5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - You must load the additional data from 'GB_category_id.json' into memory before executing any operations.
        - To access 'GB_category_id.json', use the path '/data/GB_category_id.json'.
    """
    # TODO:  Paste your code here
    vdo_df.drop_duplicates(inplace=True)

    cat_list = []
    for item in cat["items"]:
        cat_list.append((int(item["id"]), item["snippet"]["title"]))

    cat_df = pd.DataFrame(cat_list, columns=["id", "category"])
    vdo_df_with_cat = vdo_df.merge(cat_df, left_on="category_id", right_on="id")
    filtered_df = vdo_df_with_cat[
        vdo_df_with_cat["category"].isin(["Sports", "Comedy"])
    ]
    grouped_df = (
        filtered_df.groupby(["trending_date", "category"])["views"].sum().reset_index()
    )

    pivot_df = grouped_df.pivot(
        index="trending_date", columns="category", values="views"
    )
    pivot_df["sports_greater"] = pivot_df["Sports"] > pivot_df["Comedy"]
    count = pivot_df["sports_greater"].sum()

    return count


# print(Q5(vdo_df))
