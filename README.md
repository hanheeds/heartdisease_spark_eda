# Workflow Orchestration
This project aims to perform web scraping, data gathering, machine learning, and workflow orchestration for Exploratory Data Analysis (EDA) and EDA with Spark. The tasks involved are described below along with the necessary instructions and requirements.

## Prerequisites and Setup
- Python 3 installed on your system.
- Required Python packages/libraries (pandas, scrapy, scikit-learn, PySpark, etc.) installed.
- Access to AWS services (EC2, S3, RDS, etc.) for executing the workflow.
To set up the project, follow these steps:

1. Clone the project repository from [repository URL].
2. Install the required Python packages using the command: pip install -r requirements.txt.
3. Configure AWS credentials and ensure necessary permissions for accessing required services.
4. Set up Airflow on AWS EC2 or any other server/VM following the official documentation.
5. Set up the backend database (RDS, MySQL, etc.) for Airflow.

## AWS Setup

### Amazon S3
Amazon S3 is a scalable cloud storage service provided by Amazon Web Services (AWS). It is designed to store and retrieve large amounts of data from anywhere on the web.

1. Create an S3 bucket to store all of the necessary data
2. The contents of the bucket should look like the following:
  a. DAGS/
  b. requirements.txt

The DAGS/ folder should contain the pipeline.py script for creating the workflow and DAGs.

These are the packages present in our requirements.txt file:
```
apache-airflow
beautifulsoup4>= 4.9.3
matplotlib==3.7.1
numpy==1.22.4
pandas
pyspark
Requests
scikit_learn
pymysql
scitkit-learn>=1.2
boto3
s3fs<=0.4s
```

### Amazon MWAA
Managed Workflows for Apache Airflow (MWAA) is a managed orchestration service on Amazon that makes Apache Airflow easier to setup and operate. 

1. Navigate to Amazon AWS MWAA. 
2. Create an MWAA Airflow environment (this can tak 20-40 minutes) 
3. Specify the correct DAG folder and requirements.txt
4. Once the environment is available, open the Airflow UI of the environment.

## Workflow Steps for Pipeline.py
1. Load Data: This step loads or accesses the source data required for EDA.
2. Standard EDA: Executes the standard EDA using scikit-learn on the loaded data.
3. EDA with Spark: Performs EDA using Spark on the loaded data.
4. Feature Engineering 1: Applies feature engineering strategy 1 on the processed data.
5. Feature Engineering 2: Applies feature engineering strategy 2 on the processed data.
6. Each feature engineering is processed on both SVM and LR machine learning algorithms.
7. Best model: Out of the 4 models, the model with the highest accuracy is chosen.
8. Final evaluation: This final model is then evaluated on the test set along with the results.

## Running the Workflow
Once you are in the Airflow UI of the environment, follow the steps. 
1. Select the correct DAG in the dashboard and unpause the DAG. 
2. Select the DAG and navigate towards the 'Graph' tab. 
3. You will know see the DAG graph. 
4. Now on the top right hand corner, press on the play button to Trigger Dag.
5. This will allow you to run the DAG. 

<a data-flickr-embed="true" data-footer="true" href="https://www.flickr.com/photos/198446312@N02/52945464452/in/dateposted-public/" title="AirFlow Dashboard"><img src="https://live.staticflickr.com/65535/52945464452_c691fb747d_k.jpg" width="1870" height="642" alt="AirFlow Dashboard"/></a>
*Dashboard*

<a data-flickr-embed="true" href="https://www.flickr.com/photos/198446312@N02/52946067116/in/dateposted-public/" title="DAG Airflow"><img src="https://live.staticflickr.com/65535/52946067116_66232bbb9c_c.jpg" width="800" height="507" alt="DAG Airflow"/></a>
*Airflow Graph*

If there are any bugs, click on each node of the DAG and press 'log' to be able to debug.

## License
Feel free to customize and adapt this project according to your needs. If you have any questions or issues, please don't hesitate to reach out.

Enjoy exploring and analyzing your data with Airflow and Spark!
