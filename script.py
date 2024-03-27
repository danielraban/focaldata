import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ast import literal_eval


def load_csv_files(query_csv_path, dataset_csv_path):
    """
    Load query and dataset CSV files into pandas DataFrames.
    """
    query_df = pd.read_csv(query_csv_path)
    dataset_df = pd.read_csv(dataset_csv_path)
    return query_df, dataset_df


def preprocess_vector_embeddings(df, column_name="vector_embedding"):
    """
    Convert string representations of vector embeddings to numpy arrays.
    """
    df[column_name] = df[column_name].apply(lambda x: np.array(literal_eval(x)))
    return df


def filter_by_age(dataset_df, min_age, max_age):
    """
    Filter the dataset DataFrame based on age criteria.
    """
    return dataset_df[(dataset_df["age"] >= min_age) & (dataset_df["age"] <= max_age)]


def find_top_responses(query_embedding, dataset_df, top_n=10):
    """
    Find the top N most relevant responses based on cosine similarity.
    """
    similarities = cosine_similarity(
        [query_embedding], np.vstack(dataset_df["vector_embedding"])
    )
    dataset_df["similarity"] = similarities[0]
    top_responses = dataset_df.sort_values(by="similarity", ascending=False).head(top_n)
    return top_responses


def main(query_csv_path, dataset_csv_path):
    """
    Main function to load data, preprocess, filter by age, and find top responses.
    """
    # Load CSV files
    query_df, dataset_df = load_csv_files(query_csv_path, dataset_csv_path)

    # Preprocess vector embeddings
    query_df = preprocess_vector_embeddings(query_df)
    dataset_df = preprocess_vector_embeddings(dataset_df)

    # Extract query details
    query_details = query_df.iloc[0]
    query_embedding = query_details["vector_embedding"]
    min_age = query_details["min_age"]
    max_age = query_details["max_age"]

    # Filter dataset by age
    filtered_dataset = filter_by_age(dataset_df, min_age, max_age)

    # Find top responses
    top_responses = find_top_responses(query_embedding, filtered_dataset)

    # Print top responses
    for index, row in top_responses.iterrows():
        print(row["response_text"])


if __name__ == "__main__":
    # Example usage
    query_csv_path = "interview_query_movies.csv"
    dataset_csv_path = "interview_dataset_movies.csv"
    main(query_csv_path, dataset_csv_path)
