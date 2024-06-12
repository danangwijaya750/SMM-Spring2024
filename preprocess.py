import pandas as pd
from sklearn import  metrics, preprocessing
import numpy as np
import json
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from transformers import pipeline
from collections import Counter
import random
import scipy.sparse as sp

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

def classify_long_review_pipeline(review,classification):
    # Define chunk size and stride
    chunk_size = 512
    stride = 256

    # Helper function to classify a single chunk
    def classify_chunk(chunk):
        result = classification([chunk])
        if result:
            result = result[0]
            score = result.get('score', 0)  # Extract the score from the dictionary or default to 0
            label = result.get('label', 'UNKNOWN')  # Extract the label from the dictionary or default to 'UNKNOWN'
            return score, label
        return 0, 'UNKNOWN'

    # Check if the review length is shorter than the chunk size
    if len(review) <= chunk_size:
        return classify_chunk(review)

    # Split the review into overlapping chunks
    chunks = []
    for i in range(0, len(review), stride):
        chunk = review[i:i + chunk_size]
        if len(chunk) < chunk_size and i + chunk_size > len(review):
            chunk = review[i:]  # Take the remaining part of the review
        chunks.append(chunk)

    # Classify each chunk using the sentiment analysis pipeline
    chunk_results = []
    for chunk in chunks:
        score, label = classify_chunk(chunk)
        chunk_results.append((score, label))

    # Check if chunk_results is empty
    if not chunk_results:
        return None  # Return None if no chunks were generated or no scores were returned

    # # Calculate final score and determine final label
    # weighted_sum = sum(label_to_value[label] * score for score, label in chunk_results)
    # total_weight = sum(score for score, label in chunk_results)
    # final_score = weighted_sum / total_weight if total_weight != 0 else 0

    # # Determine the final label based on the sign of the final_score
    # final_label = 'POSITIVE' if final_score > 0 else 'NEGATIVE'
    total_score = sum(score for score, label in chunk_results)
    final_score = total_score / len(chunk_results)

    # Determine the most frequent label
    labels = [label for score, label in chunk_results]
    label_counter = Counter(labels)
    final_label = label_counter.most_common(1)[0][0]

    return final_score, final_label


def preprocess_sentiment_score(ratings_df):
    model_name="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    classification = pipeline('sentiment-analysis', model=model_name)
    reviews=ratings_df['text'].to_numpy()

    reviews_scores = []
    for review in tqdm(reviews, desc="Processing review sentiment score", unit=" review"):
        reviews_scores.append(classify_long_review_pipeline(review, classification))

    ratings_df['sentiment_score'] = [x[0] for x in reviews_scores]
    ratings_df['sentiment_label'] = [x[1] for x in reviews_scores]
    label_to_weight = {'negative': 0.5, 'neutral': 1.0, 'positive': 1.5}
    # Scale sentiment scores based on weights
    ratings_df['weighted_sentiment_score'] = ratings_df.apply(
        lambda row: row['sentiment_score'] * label_to_weight[row['sentiment_label']], axis=1)
    # Normalize the weighted sentiment scores to the range 0.00 to 5.00
    min_score = ratings_df['weighted_sentiment_score'].min()
    max_score = ratings_df['weighted_sentiment_score'].max()
    ratings_df['normalized_weighted_sentiment_score'] = ratings_df['weighted_sentiment_score'].apply(
        lambda x: 5.00 * (x - min_score) / (max_score - min_score))
    ratings_df['normalized_weighted_sentiment_score'] = ratings_df['normalized_weighted_sentiment_score'].apply(
        lambda x: round(x * 2) / 2 )
    return ratings_df


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
    src = [user_id for user_id in  df[src_index_col]]
    # get movie_id from rating events in the order of occurance
    dst = [(place_id) for place_id in df[dst_index_col]]

    # apply rating threshold
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    return torch.LongTensor(edge_index)

def data_split(edge_index):
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]
    train_indices, test_indices = train_test_split(all_indices,
                                               test_size=0.2,
                                               random_state=123)
    val_indices, test_indices = train_test_split(test_indices,
                                             test_size=0.5,
                                             random_state=123)
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    return train_edge_index, val_edge_index, test_edge_index, num_interactions

def handle_empty_review(ratings_df):
  for index, row in ratings_df.iterrows():
    if len(row['text'])==0:
      rating = row['rating']

      positive_keywords = ['good', 'great', 'awesome', 'love', 'recommend']
      negative_keywords = ['bad', 'terrible', 'awful', 'waste', 'disappointed']

      if rating >= 4:
        sentiment = 'positive'
      elif rating <= 2:
        sentiment = 'negative'
      else:
        sentiment = 'neutral'

      if sentiment == 'positive':
        review = f"I really enjoyed this hiking. It was {positive_keywords[random.randint(0, len(positive_keywords) - 1)]}."
      elif sentiment == 'negative':
        review = f"I was disappointed with this hiking. It was {negative_keywords[random.randint(0, len(negative_keywords) - 1)]}."
      else:
        review = "This hiking was okay."

      ratings_df.at[index, 'text'] = review

  return ratings_df


#NGCF preprocess
class DataLoader:
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.n_users, self.n_items, self.n_data = 0, 0, 0
        self.users = []
        self.pos_items = {}
        self.load()

    def load(self):
        for _, row in self.df.iterrows():
            uid, iid = int(row['user_id']), int(row['place_id'])
            if uid not in self.pos_items:
                self.pos_items[uid] = []
            self.pos_items[uid].append(iid)
            self.n_items = max(self.n_items, iid)
            self.n_users = max(self.n_users, uid)
            self.n_data += 1
            self.users.append(uid)
        self.n_users += 1
        self.n_items += 1

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u in self.users:
            for i in self.pos_items[u]:
                self.R[u, i] = 1.

    def compute_norm_adj_matrix(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def compute_adj_matrix(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32).tolil()
        A[:self.n_users, self.n_users:] = self.R.tolil()
        A[self.n_users:, :self.n_users] = self.R.tolil().T
        A = A.todok()
        mean_adj = self.compute_norm_adj_matrix(A)
        return mean_adj + sp.eye(mean_adj.shape[0])
  
    def set_adj_matrix(self):
          adj = self.compute_adj_matrix()
        #   sp.save_npz('adj_matrix.npz', adj.tocsr())
          self.adj_matrix = adj

    def sample_pos(self, u, amount):
        high = len(self.pos_items[u])
        pos_sample = []
        while len(pos_sample) < amount:
            id = np.random.randint(low=0, high=high, size=1)[0]
            item = self.pos_items[u][id]
            if item not in pos_sample:
                pos_sample.append(item)
        return pos_sample

    def sample_neg(self, u, amount):
        high = self.n_items
        neg_sample = []
        while len(neg_sample) < amount:
            item = np.random.randint(low=0, high=high, size=1)[0]
            if item not in self.pos_items[u] and item not in neg_sample:
                neg_sample.append(item)
        return neg_sample

    def sample(self):
        users = np.random.choice(self.users, size=self.batch_size)
        pos_sample, neg_sample = [], []
        for u in users:
            pos_sample += self.sample_pos(u, 1)
            neg_sample += self.sample_neg(u, 1)
        return users, pos_sample, neg_sample

def preprocess_ngcf(data,threshold):
    data.drop(data[data['rating'] < threshold].index, inplace=True)
    train_df, val_test_df = train_test_split(data, test_size=0.2, random_state=123)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=123)
    train_data = DataLoader(train_df, batch_size=32)
    val_data = DataLoader(val_df, batch_size=32)
    test_data = DataLoader(test_df, batch_size=32)
    train_data.set_adj_matrix()
    return train_data,test_data