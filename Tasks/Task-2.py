import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv(r"C:/Users/Sanjay/Desktop/Task-2/Dataset .csv")
df.rename(columns={
    'Restaurant Name': 'Restaurant_Name',
    'Cuisines': 'Cuisine',
    'Average Cost for two': 'Price_Range',
    'Aggregate rating': 'Rating'
}, inplace=True)
df['Cuisine'].fillna('Unknown', inplace=True)
df['Price_Range'].fillna(df['Price_Range'].median(), inplace=True)
df['Rating'].fillna(df['Rating'].mean(), inplace=True)
features = df[['Cuisine', 'Price_Range', 'Rating']]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Changed to sparse_output=False
encoded_cuisine = encoder.fit_transform(features[['Cuisine']])

encoded_df = pd.DataFrame(
    encoded_cuisine,
    columns=encoder.get_feature_names_out(['Cuisine'])
)

final_features = pd.concat(
    [encoded_df, features[['Price_Range', 'Rating']].reset_index(drop=True)],
    axis=1
)
similarity_matrix = cosine_similarity(final_features)
def recommend_restaurants(user_cuisine, user_price, user_rating, top_n=5):
    """
    Recommend restaurants based on user preferences
    """

    user_data = pd.DataFrame({
        'Cuisine': [user_cuisine],
        'Price_Range': [user_price],
        'Rating': [user_rating]
    })

    user_encoded = encoder.transform(user_data[['Cuisine']])

    user_vector = np.concatenate(
        [user_encoded, user_data[['Price_Range', 'Rating']].values],
        axis=1
    )

    similarity_scores = cosine_similarity(user_vector, final_features)[0]

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    return df.iloc[top_indices][
        ['Restaurant_Name', 'Cuisine', 'Price_Range', 'Rating']
    ]
if _name_ == "_main_":
    recommendations = recommend_restaurants(
        user_cuisine='Italian',
        user_price=800,      # average cost for two
        user_rating=4.0,
        top_n=5
    )

    print("\nRecommended Restaurants:\n")
    print(recommendations)