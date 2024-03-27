import unittest
from script import filter_by_age, find_top_responses  # Assume these functions are in `script.py`
import pandas as pd
import numpy as np

class TestScript(unittest.TestCase):
    def setUp(self):
        # Setup mock data
        self.dataset_df = pd.DataFrame({
            'respondent_id': [1, 2, 3, 4, 5],
            'respondent_name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
            'country': ['US', 'UK', 'CA', 'US', 'UK'],
            'age': [25, 30, 35, 40, 45],
            'response_text': ['Answer 1', 'Answer 2', 'Answer 3', 'Answer 4', 'Answer 5'],
            'vector_embedding': [np.array([0.1, 0.2]), np.array([0.2, 0.3]), np.array([0.3, 0.4]), np.array([0.4, 0.5]), np.array([0.5, 0.6])]
        })
        self.query_embedding = np.array([0.25, 0.35])

    def test_filter_by_age(self):
        # Test the age filtering function
        filtered_df = filter_by_age(self.dataset_df, 30, 40)
        self.assertEqual(len(filtered_df), 3)  # Expect 3 entries within age range 30-40

    def test_find_top_responses(self):
        # Test the function that finds top responses based on cosine similarity
        filtered_df = filter_by_age(self.dataset_df, 25, 45)  # Use wider age range for this test
        top_responses = find_top_responses(self.query_embedding, filtered_df, top_n=3)
        self.assertEqual(len(top_responses), 3)  # Expect 3 top responses
        # Further tests can assert the correctness of the selected top responses

if __name__ == '__main__':
    unittest.main()