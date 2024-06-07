import pandas as pd
from sklearn import  metrics, preprocessing
import numpy as np
import json
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

def load_encode_data(data):
    # Create bipartite graph
    ratings_data = []
    for place in tqdm(data, desc='Creating graph'):
        place_id = place['place_id']
        for review in place['reviews']:
            user_url = review['author_url']
            rating = review['rating']
            review = review['text']
            ratings_data.append([user_url, place_id, rating,review])

    ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'place_id', 'rating','text'])
    lbl_user = preprocessing.LabelEncoder()
    lbl_place = preprocessing.LabelEncoder()
    ratings_df.user_id = lbl_user.fit_transform(ratings_df.user_id.values)
    ratings_df.place_id = lbl_place.fit_transform(ratings_df.place_id.values)
    num_users = len(ratings_df['user_id'].unique())
    num_places = len(ratings_df['place_id'].unique())
    
    return ratings_df,num_users,num_places

def load_edge(df,
                  src_index_col,
                  dst_index_col,
                  link_index_col,
                  rating_threshold=3):
    """Loads csv containing edges between users and items

    Args:
        src_index_col (str): column name of users
        dst_index_col (str): column name of items
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 3.

    Returns:
        list of list: edge_index -- 2 by N matrix containing the node ids of N user-item edges
        N here is the number of interactions
    """

    edge_index = None

    # Constructing COO format edge_index from input rating events

    # get user_ids from rating events in the order of occurance
    src = [user_id for user_id in  df['user_id']]
    # get movie_id from rating events in the order of occurance
    dst = [(place_id) for place_id in df['place_id']]

    # apply rating threshold
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    return edge_index

def data_split(edge_index):
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]
    train_indices, test_indices = train_test_split(all_indices,
                                               test_size=0.2,
                                               random_state=1)
    val_indices, test_indices = train_test_split(test_indices,
                                             test_size=0.5,
                                             random_state=1)
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    return train_edge_index, val_edge_index, test_edge_index, num_interactions