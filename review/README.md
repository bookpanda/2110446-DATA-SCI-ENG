# Visualization
## Matplotlib
2 styles: pyplot and OOP (should not mix)
### pyplot
```python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
plt.plot(x, y)
plt.title("Simple Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
```

### OOP
```python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Simple Plot")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
plt.show()

# subplots: fig=box of all plots, each axis=1 graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
```

## Plotly
- based on JS, interactive
- 2 styles: Graph Objects Interface (more customizable), Plotly Express (high-level), don't mix
### Components
- Data = list of traces (go.Scatter, go.Bar, etc.)
- Layout = dict of layout properties (title, xaxis, yaxis, etc.)
- Figures (go.Figure) = data + layout

## Network
### Random Network
- node degree graph will be normal distribution

### Small-world Network
- high clustering, short path length
- social connection, email, telephone calls

### Scale-free Network
- node degree graph will be power law distribution (exponential decrease)
- most nodes: low degree, few nodes: high degree (hubs)
- social network (famous people), citation

### Centrality
- Degree Centrality: important node = high degree
- Closeness Centrality: important node = short path to all other nodes
- Betweenness Centrality: important node = high betweenness (is in most shortest paths between nodes)
- Eigenvector Centrality: important node = connected to other important nodes

# Data Engineering
## Redis
```py
rd = redis.Redis(host='lab.aimet.tech', charset="utf-8", decode_responses=True)
rd.auth(username='hw', password='hw')

# list all keys
rd.keys()

# scan for keys beginning with "user" (get no. of users), 100 at a time
cursor = 0
keys = []
while True:
    cursor, batch = r.scan(cursor=cursor, match='user*', count=100)
    keys.extend(batch)
    if cursor == 0:
        break
print([k.decode() for k in keys])  # decode from bytes to str

# username of user id "600"
rd.get('user:600:name')
# id of username "excitedPie4"
rd.get("username:excitedPie4")
# follows count
rd.scard("user:567:follows")

# avg no. of follows per user
cursor = 0
follow_count = 0
while True:
    cursor, keys = rd.scan(cursor=cursor, match='user:*:follows', count=100)
    print(cursor, keys)
    for key in keys:
        follow_count += rd.scard(key)

    if cursor == 0:
        break
print(f"Average number of follows per user: {follow_count/user_count}")

# users follows between 5 and 10
cursor = 0
user_5_10 = 0
while True:
    cursor, keys = rd.scan(cursor=cursor, match='user:*:follows', count=100)
    print(cursor, keys)
    for key in keys:
        follows = rd.scard(key)
        if follows >= 5 and follows <= 10:
            user_5_10 += 1

    if cursor == 0:
        break
print(f"No. of users that follow 5-10 users: {user_5_10}")

# user with most followers
cursor = 0
max_followers = 0
user_with_most_followers = None
while True:
    cursor, keys = rd.scan(cursor=cursor, match='user:*:followed_by', count=100)
    print(cursor, keys)
    for key in keys:
        follows = rd.scard(key)

        if follows > max_followers:
            max_followers = follows
            user_with_most_followers = key

    if cursor == 0:
        break
print(f"User with the most followers: {user_with_most_followers} with {max_followers} followers")
```

## Spark
```py
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

spark = SparkSession.builder\
        .master(spark_url)\
        .appName('Spark Tutorial')\
        .config('spark.ui.port', '4040')\
        .getOrCreate()

df_imdb = spark.read.csv("netflix-rotten-tomatoes-metacritic-imdb.csv", header=True, inferSchema=True)
cols = [c.replace(' ', '_') for c in df_imdb.columns]
df_imdb = df_imdb.toDF(*cols)
df_imdb.printSchema()
# show 5 rows
df_imdb.show(5)

from pyspark.sql.functions import avg, min, max
df_imdb.select(max("Hidden_Gem_Score"), avg("Hidden_Gem_Score")).show()

# how many movies are in Korean?
df_imdb.filter(df_imdb["Languages"].like("%Korea%")).count()

# director with highest avg hidden gem score
df_imdb.groupby("Director") \
    .agg(avg("Hidden_Gem_Score").alias("Avg_Hidden_Gem_Score")) \
    .sort("Avg_Hidden_Gem_Score", ascending=False) \
    .show()

# how many genres are there?
from pyspark.sql import functions as F
all_genres = df_imdb.select(
    F.explode(
        F.split(F.col("Genre"), ",")
    ).alias("Genre")
)
all_genres.show()
all_genres.select(F.trim(F.col("Genre"))).distinct().count()

# replace column names with underscores
cols = [c.replace(' ', '_') for c in df.columns]
df = df.toDF(*cols)

df.select(df['Date'], df['AveragePrice']).show(5)
df.select(df['Date'], df['Small_Bags']+df['Large_Bags']).show(5)
df.filter(df['Total_Bags'] < 8000).show(3)
df.filter((df['Total_Bags'] < 8000) & (df.year > 2015)).select('Date', 'Total_Bags').show(3)
df.filter('Total_Bags < 8000 and year > 2015').select('Date', 'Total_Bags').show(3)
df.select('type').distinct().show()
from pyspark.sql.functions import avg, min, max
df.select(min('AveragePrice'), avg('AveragePrice'), max('AveragePrice')).show()
df.groupby('type').count().show()
df.groupby('year', 'type').agg({'AveragePrice': 'avg'}).orderBy('year', 'type').show()

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
def pricegroup_mapping(price):
    if price < 1:
        return 'cheap'
    if price < 2:
        return 'moderate'
    return 'expensive'
to_pricegroup = udf(pricegroup_mapping, StringType())
df.select('Date', 'AveragePrice', to_pricegroup('AveragePrice')).show(5)
new_df = df.withColumn('pricegroup', to_pricegroup(df.AveragePrice))
new_df.select('AveragePrice', 'pricegroup').show(5)
```

## Kafka
```py
from kafka import KafkaConsumer, KafkaProducer
import avro.schema
import avro.io
import io
import hashlib, json
import threading
import time

def serialize(schema, obj):
    bytes_writer = io.BytesIO()
    encoder = avro.io.BinaryEncoder(bytes_writer)
    writer = avro.io.DatumWriter(schema)
    writer.write(obj, encoder)
    return bytes_writer.getvalue()

def deserialize(schema, raw_bytes):
    bytes_reader = io.BytesIO(raw_bytes)
    decoder = avro.io.BinaryDecoder(bytes_reader)
    reader = avro.io.DatumReader(schema)
    return reader.read(decoder)

schema_file = 'transaction.avsc'
txschema = avro.schema.parse(open(schema_file).read())
schema_file = 'submit.avsc'
submitschema = avro.schema.parse(open(schema_file).read())
schema_file = 'result.avsc'
resultschema = avro.schema.parse(open(schema_file).read())
# transaction.avsc
{
   "namespace": "assignment.avro",
   "type": "record",
   "name": "transaction",
   "fields": [
      {"name": "txid", "type": "string"},
      {"name": "payer", "type": "string"},
      {"name": "payee", "type": "string"},
      {"name": "amount", "type": "int"}
   ] 
 }

kafka_broker = 'lab.aimet.tech:9092'
producer = KafkaProducer(bootstrap_servers=[kafka_broker])

txconsumer = KafkaConsumer(
    'transaction',
     bootstrap_servers=[kafka_broker],
     enable_auto_commit=True,
     value_deserializer=lambda x: deserialize(txschema, x))
resultconsumer = KafkaConsumer(
    'result',
     bootstrap_servers=[kafka_broker],
     enable_auto_commit=True,
     value_deserializer=lambda x: deserialize(resultschema, x))

def gen_signature(txid, payer, payee, amount, token):
    o = {'txid': txid, 'payer': payer, 'payee': payee, 'amount': amount, 'token': token}
    return hashlib.md5(json.dumps(o, sort_keys=True).encode('utf-8')).hexdigest()

def monitor_thread():
    for message in resultconsumer:
        print(message.value)
monitor = threading.Thread(target=monitor_thread, daemon=True)
monitor.start()

print('Running TX Consumer')
for message in txconsumer:
    txid = message.value['txid']
    payer = message.value['payer']
    payee = message.value['payee']
    amount = message.value['amount']

    signature = gen_signature(txid, payer, payee, amount, token)
    submit_data = {'vid': vid, 'txid': txid, 'signature': signature}
    serialized_submit_data = serialize(submitschema, submit_data)

    producer.send('submit', serialized_submit_data)
    print(message.value)
    print(f"submit: {submit_data}")
```

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

## Data Prep
kinds of data:
1. Categorical: nominal (no order), ordinal (order)
2. Numerical: discrete (countable), continuous (infinite values)

### Unqalified features
- IDs
- missing values > 50%
- categorical vars
    - too many unique values (becomes ID) -> grouping
    - flat values (underfit)
- zip code -> distance to closest branch

### Impute
- numerical: mean, median
- categorical: mode
- group stats e.g. income by age group

### Categorical to numerical
- ordinal: enumerate (bachelor=1, master=2, phd=3)
- nominal: one-hot (100, 010, 001), dummy codes (10, 01, 00), target avg, weight of evidence

### Feature Transformation
- log
- binning

K-fold cross-validation
- fixes overfitting on test
- split data into k folds
- train on k-1 folds, test on 1 fold
- repeat k times, each time using a different fold as test set

### Data Leakage Sols
- split by subjects/videos rather than individual images
- for time series, split by time

## Classification
- Naive Bayes
- Logistic Regression
- Random Forest
- Gradient Boosting (many Decision Trees trained independently)
- XGBoost (Gradient Boosting with regularization, optimization)
- k-Nearest Neighbors

## Regression
- Linear Regression
- Decision Tree
- Support Vector Regression

## Linear Regression
y = Xw + b
### Assumptions
- must do standard scaling for all features
- linear relationship
- error is normally distributed (remove outliers, log transform)
- error has equal variance (remove outliers, log transform)
- error are independent
### Regularization
- Linear regression: MSE
- Ridge loss: MSE + squared error (L2)
  - when there are many small/colinear features
- Lasso loss: MSE + absolute error (L1)
  - expect few important features
- ElasticNet: MSE + aL1 + bL2
```py
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model_lr = LinearRegression()
model_ridge = Ridge(alpha=1.0)
model_lasso = Lasso(alpha=0.1)
model_elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)

model_lr.fit(X, y)
model_ridge.fit(X, y)
model_lasso.fit(X, y)
```

## Logistic Regression
### Assumptions
- must do standard scaling for all features
- mean of logit is linear
  - (logit, x) = linear relationship
- errors are independent

## k Nearest Neighbors
- can use GridSearchCV to find best k

## Random Forest
```py
# clean
hw.dropna(subset=['label'], inplace=True)
hw.drop(columns=['id','gill-attachment', 'gill-spacing'], inplace=True)
hw.reset_index(inplace=True)
```
