from airflow import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import when, rand

from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
from io import StringIO
from bs4 import BeautifulSoup
import requests
import boto3

WORKFLOW_SCHEDULE_INTERVAL = None
BUCKET_NAME = 'de300-group7-mwaa-output'

default_args = {
    'owner': 'group7',
    'depends_on_past': False,
    'start_date': days_ago(1)
}

spark = SparkSession.builder.appName("de300-project-module3").getOrCreate()

def load(**kwargs):

    s3hook = S3Hook(aws_conn_id='aws_conn_id')
    bucket = "de300-group7-module3"
    key = "heart_disease.csv"
    s3data = s3hook.read_key(bucket_name=bucket, key=key)

    # make df
    data = pd.read_csv(StringIO(s3data))
    # get rid of bottom rows that are erroneous
    df = data[0:899]
    df.reset_index(inplace=True)

    # now in spark
    spark_df = spark.createDataFrame(df)

    # sample for test key 
    test_indices = np.array(df.sample(frac=0.1, random_state=42)['index'])

    kwargs['ti'].xcom_push(key='df',value=df)
    kwargs['ti'].xcom_push(key='spark_df',value=spark_df)
    kwargs['ti'].xcom_push(key='test_indices',value=test_indices)


def clean1(**kwargs):
    '''
        Removes columns missing more than 10% of the data, and imputes missing values using 1 nearest neighbor.
    '''
    df = kwargs['ti'].xcom_pull(key='df')

    # Remove columns with more than 10% of data missing

    # Get the counts of null for each column
    null_count = df.isnull().sum()

    # If null counts are higher than 10%, we will drop the column. 
    cols_to_drop = null_count[null_count > int(df.shape[0]*0.1)].index.tolist()

    # Drop the columns 
    df = df.drop(columns=cols_to_drop)

    # fix dtypes
    df = df.convert_dtypes()
    df['age'] = df['age'].astype('Int64')

    columns = df.columns

    # Impute

    # impute mising values using kNN
    imputer = KNNImputer(n_neighbors=1)
    data = imputer.fit_transform(df)
    # back into df
    data = pd.DataFrame(data,columns=columns)
    # fix dtypes
    data = data.convert_dtypes()

    kwargs['ti'].xcom_push(key='df',value=data)
    
def clean2(**kwargs):
    '''
        Retains certain columns and imputes missing values
    '''
    df = kwargs['ti'].xcom_pull(key='spark_df')

    # 1. Retain only the following columns (apart from target)
    heart_data = df.select("age", "sex", "painloc", "painexer", "cp", "trestbps", "smoke", "fbs", "prop", "nitr", "pro", "diuretic", "thaldur", "thalach", "exang", "oldpeak", "slope", "target")

    # 2. Cleaning and imputing steps for columns other than `smoke`

    # a. painloc, painexer: Replace the missing values with most common class
    imputer = Imputer(strategy='mode', missingValue=None,inputCols=["painloc","painexer"],outputCols=["painloc","painexer"])
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)

    # b. trestbps: Replace values less than 100 mm Hg, and the missing values
    # replace values less than 100 mm Hg with null
    heart_data = heart_data.withColumn('trestbps', when(heart_data['trestbps'] >= 100,heart_data['trestbps']))
    # impute those and missing with mean
    imputer = Imputer(strategy='mean', missingValue=None,inputCol="trestbps",outputCol="trestbps")
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)

    # c. oldpeak: Replace values less than 0, those greater than 4, and the missing values
    # replace values less than 0, those greater than 4
    heart_data = heart_data.withColumn('oldpeak', when(heart_data['oldpeak'] >= 0,heart_data['oldpeak']))
    heart_data = heart_data.withColumn('oldpeak', when(heart_data['oldpeak'] <= 4,heart_data['oldpeak']))
    # impute those and missing with mean
    imputer = Imputer(strategy='mean', missingValue=None,inputCol="oldpeak",outputCol="oldpeak")
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)

    # d. thaldur, thalach: Replace the missing values with mean
    imputer = Imputer(strategy='mean', missingValue=None,inputCols=["thaldur","thalach"],outputCols=["thaldur","thalach"])
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)

    # e. fbs, prop, nitr, pro, diuretic: Replace the missing values and values greater than 1
    # replace values greater than 1
    heart_data = heart_data.withColumn('fbs', when(heart_data['fbs'] <= 1,heart_data['fbs']))
    heart_data = heart_data.withColumn('prop', when(heart_data['prop'] <= 1,heart_data['prop']))
    heart_data = heart_data.withColumn('nitr', when(heart_data['nitr'] <= 1,heart_data['nitr']))
    heart_data = heart_data.withColumn('pro', when(heart_data['pro'] <= 1,heart_data['pro']))
    heart_data = heart_data.withColumn('diuretic', when(heart_data['diuretic'] <= 1,heart_data['diuretic']))
    # impute with most common class 
    imputer = Imputer(strategy='mode', missingValue=None,inputCols=["fbs", "prop", "nitr", "pro", "diuretic"],outputCols=["fbs", "prop", "nitr", "pro", "diuretic"])
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)

    # f. exang, slope: Replace the missing values
    # replace exang with most common class
    imputer = Imputer(strategy='mode', missingValue=None,inputCol="exang",outputCol="exang")
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)
    # replace slope with mean
    imputer = Imputer(strategy='mean', missingValue=None,inputCol="slope",outputCol="slope")
    model = imputer.fit(heart_data)
    heart_data = model.transform(heart_data)

    kwargs['ti'].xcom_push(key='spark_df',value=heart_data)

def fe1(**kwargs) -> pd.Dataframe:
    '''
        Finds outliers and removes them. Preforms normalization to transform the data.
    '''
    df = kwargs['ti'].xcom_pull(key='df')

    # Remove outliers
    
    columns = df.columns

    unique_values = []
    for col in columns:
        unique_values.append(len(np.unique(data[col])))
    categorical_columns = [columns[i] for i, unique_count in enumerate(unique_values) if unique_count <= 4]
    numerical_columns = np.setdiff1d(columns,categorical_columns)

    data = df[df[numerical_columns].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    
    # Normalize data

    # normalize all numerical data
    for col in numerical_columns:
        tmp = data[col]
        data[col]=(tmp - tmp.mean())/tmp.std()
    
    kwargs['ti'].xcom_push(key='df',value=data)

def fe2(**kwargs):
    '''
        Applies one hot encoding to "cp" column
    '''
    df = kwargs['ti'].xcom_pull(key='spark_df')

    # Create a OneHotEncoder instance
    encoder = OneHotEncoder(inputCols=["cp"], outputCols=["cp_vec"])

    # Fit and transform the DataFrame
    encoded_df = encoder.fit(df).transform(df)

    kwargs['ti'].xcom_push(key='spark_df',value=encoded_df)

def lr1(**kwargs):

    df = kwargs['ti'].xcom_pull(key='df')
    indices = kwargs['ti'].xcom_pull(key='test_indices')

    test = df.loc[df.index[indices]]
    train = df[~df.index.isin(test.index)]

    X_train = train.drop('index', axis=1)
    X_train = X_train.drop('target', axis=1)

    X_test = test.drop('index', axis=1)
    X_test = X_test.drop('target', axis=1)

    y_train = train['target']
    y_test = test['target']

    # Create an instance of Logistic Regression model
    logreg = LogisticRegression()

    # Train the model using the training data
    logreg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = logreg.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    kwargs['ti'].xcom_push(key='lr_accuracy1',value=accuracy)
    kwargs['ti'].xcom_push(key='lr1',value=logreg)

def lr2(**kwargs):

    spark_df = kwargs['ti'].xcom_pull(key='spark_df')
    indices = kwargs['ti'].xcom_pull(key='test_indices')

    spark_df = spark_df.withColumn("age",spark_df["age"].cast('int'))
    # convert DataFrame to feature vector format
    assembler = VectorAssembler(inputCols=["age", "sex", "painloc", "painexer", "cp", "trestbps","fbs", "prop", "nitr", "pro", "diuretic","thaldur","thalach","exang","oldpeak","slope"], outputCol="features")
    feature_vector = assembler.transform(spark_df)

    test = feature_vector.filter(feature_vector['index'].isin(indices.to_list()))
    train = feature_vector.subtract(test)

    # train a logistic regression model
    log_reg = LogisticRegression(featuresCol="features", labelCol="target")

    bcEvaluator = BinaryClassificationEvaluator.setLabelCol("target")
    mcEvaluator = MulticlassClassificationEvaluator.setLabelCol("target")

    paramGrid = (ParamGridBuilder()
             .addGrid(log_reg.regParam, [0.01, 0.5, 2.0])
             .addGrid(log_reg.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(log_reg.maxIter, [1, 5, 10])
             .build()
    )

    # Create a 5-fold CrossValidator
    cv = CrossValidator(estimator=log_reg, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=5, parallelism = 1)
 
    # Run cross validations. This step takes a few minutes and returns the best model found from the cross validation.
    cvModel = cv.fit(train)

    best_lr = cvModel.bestModel

    # Use the model identified by the cross-validation to make predictions on the test dataset
    predictions = best_lr.transform(test)

    accuracy = mcEvaluator.evaluate(predictions, {mcEvaluator.metricName: "accuracy"})

    kwargs['ti'].xcom_push(key='lr_accuracy2',value=accuracy)
    kwargs['ti'].xcom_push(key='lr2',value=best_lr)

def svm1(**kwargs):
    
    df = kwargs['ti'].xcom_pull(key='df')
    indices = kwargs['ti'].xcom_pull(key='test_indices')

    test = df.loc[df.index[indices]]
    train = df[~df.index.isin(test.index)]

    X_train = train.drop('index', axis=1)
    X_train = X_train.drop('target', axis=1)

    X_test = test.drop('index', axis=1)
    X_test = X_test.drop('target', axis=1)

    y_train = train['target']
    y_test = test['target']
    # Create an instance of SVM model
    svm = SVC()

    # Train the model using the training data
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    kwargs['ti'].xcom_push(key='svm_accuracy1',value=accuracy)
    kwargs['ti'].xcom_push(key='svm1',value=svm)

def svm2(**kwargs):

    spark_df = kwargs['ti'].xcom_pull(key='spark_df')
    indices = kwargs['ti'].xcom_pull(key='test_indices')

    spark_df = spark_df.withColumn("age",spark_df["age"].cast('int'))
    # convert DataFrame to feature vector format
    assembler = VectorAssembler(inputCols=["age", "sex", "painloc", "painexer", "cp", "trestbps","fbs", "prop", "nitr", "pro", "diuretic","thaldur","thalach","exang","oldpeak","slope"], outputCol="features")
    feature_vector = assembler.transform(spark_df)

    test = feature_vector.filter(feature_vector['index'].isin(indices.to_list()))
    train = feature_vector.subtract(test)

    # Create an instance of Linear SVM model
    svm = LinearSVC(maxIter=10, regParam=0.1)

    # Train the model using the training data
    svmModel = svm.fit(train)

    # Make predictions on the test set
    predictions = svmModel.transform(test)

    # Evaluate the model
    mcEvaluator = MulticlassClassificationEvaluator.setLabelCol("target")
    
    accuracy = mcEvaluator.evaluate(predictions, {mcEvaluator.metricName: "accuracy"})

    kwargs['ti'].xcom_push(key='svm_accuracy2',value=accuracy)
    kwargs['ti'].xcom_push(key='svm2',value=svmModel)

def web_scrape(**kwargs):
    
    heart_data = kwargs['ti'].xcom_pull(key='spark_df')

    # Imputing the smoke column

    # source 1
    # web scraping
    url = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    x = soup.body.table.tbody.find_all('tr')

    categories = [None] * len(x)
    values = [None] * len(x)
    for ii in range(len(x)):
        categories[ii] = x[ii].th.text
        values[ii] = x[ii].td.text

    # new column with rates
    heart_data = heart_data.withColumn('smoke rate source 1', when((heart_data['smoke'].isNull()) & ((heart_data['age'] >= categories[0][0:2]) & (heart_data['age'] <= categories[0][3:5])),float(values[0])).otherwise(heart_data['smoke']))
    for ii in range(1,len(x)-1):
        heart_data = heart_data.withColumn('smoke rate source 1', when((heart_data['smoke'].isNull()) & ((heart_data['age'] >= categories[ii][0:2]) & (heart_data['age'] <= categories[ii][3:5])),float(values[ii])).otherwise(heart_data['smoke rate source 1']))
    heart_data = heart_data.withColumn('smoke rate source 1', when((heart_data['smoke'].isNull()) & (heart_data['age'] >= categories[len(x)-1][0:2]),float(values[len(x)-1])).otherwise(heart_data['smoke rate source 1']))
    
    # fill missing smoke values with 0 or 1 based on random sample and rates
    heart_data = heart_data.withColumn("rand", when(heart_data['smoke'].isNull(),rand(seed=42)))
    heart_data = heart_data.withColumn("smoke source 1", when((heart_data['smoke'].isNull()) & (heart_data['rand']*100 <= heart_data["smoke rate source 1"]),1).otherwise(heart_data["smoke"]))
    heart_data = heart_data.fillna(0,subset="smoke source 1")

    # source 2
    # web scraping
    url = 'https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    men_rate = soup.body.find_all('ul')[2].find_all('li')[0].text[-6:-2]
    women_rate = soup.body.find_all('ul')[2].find_all('li')[1].text[-6:-2]

    x = soup.body.find_all('ul')[3].find_all('li')

    categories = [None] * len(x)
    values = [None] * len(x)

    categories[0] = x[0].text[-18:-13]
    categories[1] = x[1].text[-19:-14]
    categories[2] = x[2].text[-19:-14]
    categories[3] = x[3].text[-25:-23]

    values[0] = x[0].text[-5:-2]
    values[1] = x[1].text[-6:-2]
    values[2] = x[2].text[-6:-2]
    values[3] = x[3].text[-5:-2]

    # new column with rates
    # first the women
    heart_data = heart_data.withColumn('smoke rate source 2', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 0) & ((heart_data['age'] >= categories[0][0:2]) & (heart_data['age'] <= categories[0][3:5]))),float(values[0])).otherwise(heart_data['smoke']))
    for ii in range(1,len(x)-1):
        heart_data = heart_data.withColumn('smoke rate source 2', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 0) & ((heart_data['age'] >= categories[ii][0:2]) & (heart_data['age'] <= categories[ii][3:5]))),float(values[ii])).otherwise(heart_data['smoke rate source 2']))
    heart_data = heart_data.withColumn('smoke rate source 2', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 0) &  (heart_data['age'] >= categories[len(x)-1][0:2])),float(values[len(x)-1])).otherwise(heart_data['smoke rate source 2']))
    # now the men 
    for ii in range(len(x)-1):
        heart_data = heart_data.withColumn('smoke rate source 2', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 1) & ((heart_data['age'] >= categories[ii][0:2]) & (heart_data['age'] <= categories[ii][3:5]))),float(values[ii]) * float(men_rate)/float(women_rate)).otherwise(heart_data['smoke rate source 2']))
    heart_data = heart_data.withColumn('smoke rate source 2', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 1) &  (heart_data['age'] >= categories[len(x)-1][0:2])),float(values[len(x)-1]) * float(men_rate)/float(women_rate)).otherwise(heart_data['smoke rate source 2'])) 

    # fill missing smoke values with 0 or 1 based on random sample and rates
    heart_data = heart_data.withColumn("smoke source 2", when((heart_data['smoke'].isNull()) & (heart_data['rand']*100 <= heart_data["smoke rate source 2"]),1).otherwise(heart_data["smoke"]))
    heart_data = heart_data.fillna(0,subset="smoke source 2")

    # source 3
    # web scraping
    url = 'https://wayback.archive-it.org/5774/20211119144357/https://www.healthypeople.gov/2020/data-search/Search-the-Data?nid=5287'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    labels = soup.find_all(class_='ds-inner-poptitle')
    rates = soup.find_all(class_='ds-2018')
    rates = rates[1:]

    categories = []
    values = []
    for ii in range(len(labels)):
        if labels[ii].text == "Male ":
            men_rate = float(rates[ii].span.text)
        elif labels[ii].text == "Female ":
            women_rate = float(rates[ii].span.text)
        elif labels[ii].text == "18-24 years ":
            categories.append("18-24")
            values.append(float(rates[ii].span.text))
        elif labels[ii].text == "25-44 years ":
            categories.append("25-44")
            values.append(float(rates[ii].span.text))
        elif labels[ii].text == "45-54 years ":
            categories.append("45-54")
            values.append(float(rates[ii].span.text))
        elif labels[ii].text == "55-64 years ":
            categories.append("55-64")
            values.append(float(rates[ii].span.text))
        elif labels[ii].text == "65-74 years ":
            categories.append("65-74")
            values.append(float(rates[ii].span.text))
        elif labels[ii].text == "75-84 years ":
            categories.append("75+")
            values.append(float(rates[ii].span.text))

    # new column with rates
    # first the women
    heart_data = heart_data.withColumn('smoke rate source 3', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 0) & ((heart_data['age'] >= categories[0][0:2]) & (heart_data['age'] <= categories[0][3:5]))),float(values[0])).otherwise(heart_data['smoke']))
    for ii in range(1,len(x)-1):
        heart_data = heart_data.withColumn('smoke rate source 3', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 0) & ((heart_data['age'] >= categories[ii][0:2]) & (heart_data['age'] <= categories[ii][3:5]))),float(values[ii])).otherwise(heart_data['smoke rate source 2']))
    heart_data = heart_data.withColumn('smoke rate source 3', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 0) &  (heart_data['age'] >= categories[len(x)-1][0:2])),float(values[len(x)-1])).otherwise(heart_data['smoke rate source 2']))
    # now the men 
    for ii in range(len(x)-1):
        heart_data = heart_data.withColumn('smoke rate source 3', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 1) & ((heart_data['age'] >= categories[ii][0:2]) & (heart_data['age'] <= categories[ii][3:5]))),float(values[ii]) * float(men_rate)/float(women_rate)).otherwise(heart_data['smoke rate source 2']))
    heart_data = heart_data.withColumn('smoke rate source 3', when((heart_data['smoke'].isNull()) & ((heart_data['sex'] == 1) &  (heart_data['age'] >= categories[len(x)-1][0:2])),float(values[len(x)-1]) * float(men_rate)/float(women_rate)).otherwise(heart_data['smoke rate source 2'])) 

    # fill missing smoke values with 0 or 1 based on random sample and rates
    heart_data = heart_data.withColumn("smoke source 3", when((heart_data['smoke'].isNull()) & (heart_data['rand']*100 <= heart_data["smoke rate source 3"]),1).otherwise(heart_data["smoke"]))
    heart_data = heart_data.fillna(0,subset="smoke source 3")
    heart_data = heart_data.drop("rand")

    kwargs['ti'].xcom_push(key='merge_df',value=heart_data)

def merge(**kwargs):

    merge_df = kwargs['ti'].xcom_pull(key='merge_df')
    merge_df = merge_df.toPandas()

    df = kwargs['ti'].xcom_pull(key='df')

    merge_df = pd.merge(df, merge_df[['cp_vec','smoke source 1','smoke source 2','smoke source 3']], left_index=True, right_index=True)

    kwargs['ti'].xcom_push(key='merge_df',value=merge_df)

def lr3(**kwargs):

    df = kwargs['ti'].xcom_pull(key='merge_df')
    indices = kwargs['ti'].xcom_pull(key='test_indices')

    test = df.loc[df.index[indices]]
    train = df[~df.index.isin(test.index)]

    X_train = train.drop('index', axis=1)
    X_train = X_train.drop('target', axis=1)

    X_test = test.drop('index', axis=1)
    X_test = X_test.drop('target', axis=1)

    y_train = train['target']
    y_test = test['target']

    # Create an instance of Logistic Regression model
    logreg = LogisticRegression()

    # Train the model using the training data
    logreg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = logreg.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    kwargs['ti'].xcom_push(key='lr_accuracy3',value=accuracy)
    kwargs['ti'].xcom_push(key='lr3',value=logreg)

def svm3(**kwargs):
    
    df = kwargs['ti'].xcom_pull(key='merge_df')
    indices = kwargs['ti'].xcom_pull(key='test_indices')

    test = df.loc[df.index[indices]]
    train = df[~df.index.isin(test.index)]

    X_train = train.drop('index', axis=1)
    X_train = X_train.drop('target', axis=1)

    X_test = test.drop('index', axis=1)
    X_test = X_test.drop('target', axis=1)

    y_train = train['target']
    y_test = test['target']
    # Create an instance of SVM model
    svm = SVC()

    # Train the model using the training data
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    kwargs['ti'].xcom_push(key='svm_accuracy3',value=accuracy)
    kwargs['ti'].xcom_push(key='svm3',value=svm)


def find_best(**kwargs):

    lr_accuracy1 = kwargs['ti'].xcom_pull(key='lr_accuracy1')
    lr_accuracy2 = kwargs['ti'].xcom_pull(key='lr_accuracy2')
    lr_accuracy3 = kwargs['ti'].xcom_pull(key='lr_accuracy3')
    svm_accuracy1 = kwargs['ti'].xcom_pull(key='svm_accuracy1')
    svm_accuracy2 = kwargs['ti'].xcom_pull(key='svm_accuracy2')
    svm_accuracy3 = kwargs['ti'].xcom_pull(key='svm_accuracy3')

    accuracys = [lr_accuracy1, lr_accuracy2, lr_accuracy3, svm_accuracy1, svm_accuracy2, svm_accuracy3]
    
    best_accuracy = np.argmax(accuracys)

    if best_accuracy == 0:
        print('best model is sklearn lr')
        kwargs['ti'].xcom_push(key='best_model',value='lr1')
    elif best_accuracy == 1:
        print('best model is spark lr')
        kwargs['ti'].xcom_push(key='best_model',value='lr2')
    elif best_accuracy == 2:
        print('best model is merge lr')
        kwargs['ti'].xcom_push(key='best_model',value='lr3')
    elif best_accuracy == 3:
        print('best model is sklearn svm')
        kwargs['ti'].xcom_push(key='best_model',value='svm1')
    elif best_accuracy == 4:
        print('best model is spark svm')
        kwargs['ti'].xcom_push(key='best_model',value='svm2')
    elif best_accuracy == 5:
        print('best model is merge svm')
        kwargs['ti'].xcom_push(key='best_model',value='svm3')

def evaluate(**kwargs):

    best_model = kwargs['ti'].xcom_pull(key='best_model')

    s3hook = S3Hook(aws_conn_id='aws_conn_id')
    bucket = "de300-group7-module3"
    key = "heart_disease.csv"
    s3data = s3hook.read_key(bucket_name=bucket, key=key)

    # make df
    data = pd.read_csv(StringIO(s3data))
    # get rid of bottom rows that are erroneous
    df = data[0:899]

    indices = kwargs['ti'].xcom_pull(key='test_indices')

    test = df.loc[df.index[indices]]
    
    model = kwargs['ti'].xcom_pull(key=best_model)

    if best_model == 'lr2' or best_model == 'svm2':
        # Make predictions on the test set
        predictions = model.transform(test)

        # Evaluate the model
        mcEvaluator = MulticlassClassificationEvaluator.setLabelCol("target")
    
        accuracy = mcEvaluator.evaluate(predictions, {mcEvaluator.metricName: "accuracy"})
    else: 
        X_test = test.drop('target', axis=1)
        y_test = test['target']

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
    print(f"Best model accuracy is: {accuracy}")


###################################

dag = DAG(
    'Classify',
    default_args=default_args,
    description='Classify with feature engineering and model selection',
    tags=["de300"]
)

load_data = PythonOperator(
    task_id='load_data',
    python_callable=load,
    dag=dag
)

clean_module1 = PythonOperator(
    task_id='clean_module1',
    python_callable=clean1,
    dag=dag
)

clean_module2 = PythonOperator(
    task_id='clean_module2',
    python_callable=clean2,
    dag=dag
)

fe_module1 = PythonOperator(
    task_id='feature_engineering_module1',
    python_callable=fe1,
    dag=dag
)

fe_module2 = PythonOperator(
    task_id='feature_engineering_module2',
    python_callable=fe2,
    dag=dag
)

lr_module1 = PythonOperator(
    task_id='logistic_regression_module1',
    python_callable=lr1,
    dag=dag
)

lr_module2 = PythonOperator(
    task_id='logistic_regression_module2',
    python_callable=lr2,
    dag=dag
)

svm_module1 = PythonOperator(
    task_id='svm_module1',
    python_callable=svm1,
    dag=dag
)

svm_module2 = PythonOperator(
    task_id='svm_module2',
    python_callable=svm2,
    dag=dag
)

webscrape = PythonOperator(
    task_id='scrape',
    python_callable=web_scrape,
    dag=dag
)

merge_data = PythonOperator(
    task_id='merge_data',
    python_callable=merge,
    dag=dag
)

lr_merge = PythonOperator(
    task_id='lr_merge',
    python_callable=lr3,
    dag=dag
)

svm_merge = PythonOperator(
    task_id='svm_merge',
    python_callable=svm3,
    dag=dag
)

choose_best = PythonOperator(
    task_id='choose_best',
    python_callable=find_best,
    dag=dag
)

evaluate_best = PythonOperator(
    task_id='evaluate_best',
    python_callable=evaluate,
    dag=dag
)

load_data >> clean_module1 >> fe_module1 >> [lr_module1, svm_module1]
load_data >> clean_module2 >> fe_module2 >> [lr_module2, svm_module2]
[fe_module1, fe_module2, webscrape] >> merge_data >> [lr_merge, svm_merge]
[lr_module1, svm_module1, lr_module2, svm_module2, lr_merge, svm_merge] >> choose_best >> evaluate_best