# Libraries Used
- pandas (as pd): Used for loading, manipulating, and filtering data in tabular form (DataFrames).
- cosine_similarity from sklearn.metrics.pairwise: A function to calculate the cosine similarity between vectors, used here to find similarities between query and dataset embeddings.
- numpy (as np): Provides support for arrays and includes mathematical functions to operate on these arrays.
- literal_eval from ast: Safely evaluates a string containing a Python literal or container display into the corresponding Python object. Here, it's used to convert string representations of vectors back into actual vector objects (arrays).

# Loading the CSV Files
- Use pandas.read_csv to load interview_query_movies.csv and interview_dataset_movies.csv.

# Preprocessing the Data
- The vector_embedding columns are likely stored as strings that represent tuples. Convert these strings back into actual Python tuples or numpy arrays.
- This conversion is necessary for calculating cosine similarity. Use ast.literal_eval for a safe conversion if they're stored as strings.
- Filtering Responses by Age

# Extract min_age and max_age from interview_query_movies.csv.
- Filter interview_dataset_movies.csv for responses where the respondent's age is within the specified range.
- Calculate the cosine similarity between the query's vector embedding and each of the vector embeddings of the filtered responses.

# Selecting the Top 10 Responses
- Sort the filtered responses by their cosine similarity scores in descending order.
- Select the top 10 responses based on their scores.

# Output Format
- Display or return the response_text of these top 10 responses.

# Learned
- Vector embeddings are a fundamental concept in machine learning, allowing for the efficient and effective representation of complex data in a form that machine learning models can process. They enable models to learn and make predictions based on the semantic relationships inherent in the data.
- NumPy arrays are a foundational tool for numerical computing in Python, enabling efficient storage and manipulation of numerical data, and forming the basis of many Python-based scientific computing and data science libraries.
- Cosine similarity 
    - Cosine similarity is a metric used to measure how similar two vectors are, regardless of their magnitude. It is widely used in various fields such as information retrieval, text analysis, and machine learning, particularly in natural language processing (NLP) where it helps in assessing the similarity between documents or words represented as vector embeddings.
    - Text Analysis: In NLP, cosine similarity is used to measure the similarity between two text documents or sentences after they have been converted into vectors (usually TF-IDF vectors or word embeddings).
    - Recommendation Systems: It can compare user or item profiles represented as vectors to make recommendations based on similarity scores.
    - Clustering and Classification: Cosine similarity is useful in clustering algorithms to measure the similarity between data points and in classification tasks to find the closest class.

# Improvements
- Parallelism
    - Operations that are independent of each other and can be run in parallel should leverage multiprocessing (using Python's multiprocessing module) or multithreading (using concurrent.futures).
- Incremental Loading 
    - For very large files that don't fit into memory, process the data in chunks using pandas.read_csv's chunksize parameter
- Cloud Services
    - For massive datasets or distributed processing, use cloud-based services like Amazon S3 for storage, Amazon RDS or Google Cloud SQL for databases, and Amazon EC2 or Google Compute Engine for computation.

# Questions
- CSV data
    - How are the CSVs generated?
    - Can the query be dynamic
    - Could this script be used with FastAPI to take arguments and return the data on the API?
    - Who is responsible for this data?
    - Where is this data stored?
    - What other calculations can be done on this data to provide useful insights?


