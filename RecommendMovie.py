import tensorflow as tf
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from model_Init import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf

import os
import pickle
import re
from tensorflow.python.ops import math_ops

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime
import random

load_dir = load_params()

#users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
## 获取tensor
def load_data():
    """
    Load Dataset from File
    """
    #读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep=':', header=None, names=users_title, engine = 'python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    #改变User数据中性别和年龄
    gender_map = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {1: 0, 35: 1, 45: 2, 50: 3, 18: 4, 56: 5, 25: 6}
    users['Age'] = users['Age'].map(age_map)
    print (users)
    #读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
    movies_orig = movies.values
    #将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    #电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val:ii for ii, val in enumerate(genres_set)}

    #将电影类型转成等长数字列表，长度是18
    genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])
    
    movies['Genres'] = movies['Genres'].map(genres_map)

    #电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    
    title_set.add('<PAD>')
    title2int = {val:ii for ii, val in enumerate(title_set)}

    #将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}
    
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
    
    movies['Title'] = movies['Title'].map(title_map)

    #读取评分数据集
    ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    #合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)
    
    #将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    
    features = features_pd.values
    features=np.delete(features, [0,1], axis=1)

    targets_values = targets_pd.values
    pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig), open('preprocess.p', 'wb'))
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig

def getUserMovie(user_id_val): #[2791 'Airplane! (1980)' 'Comedy' 4]
    #title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))
    #title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
    temp=ratings.values[np.where(ratings.values[:,0]==user_id_val)]
    #print(temp)
    result=[]
    for val in temp:
        a=movies_orig[np.where(movies_orig[:,0]==val[1])]
        for x in a:
            result.append(x)
    final_result=np.insert(np.array(result),3,values=temp[:,2],axis=1)
    movie5=[]
    movie4=[]
    for i in range(len(final_result)):
        if final_result[i][3]==5:
            movie5.append(final_result[i][0])
        elif final_result[i][3]==4:
            movie4.append(final_result[i][0])
    return final_result,movie5,movie4

def newUser(gender,age,occupation):
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep=':', header=None, names=users_title, engine = 'python')
    userID=len(users)+1
    temp=pd.DataFrame({'UserID':[userID], 'Gender':[gender],'Age': [age], 'JobID':[occupation], 'Zip-code':['00000']})
    users = users.append(temp, ignore_index=True)
    users.to_csv('./ml-1m/users.dat',index=False,header=0,sep=':')
    return userID
  
def get_tensors(loaded_graph):

    #uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    #movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    #两种不同计算预测评分的方案使用不同的name获取tensor inference
#     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
    inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0") # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return user_gender, user_age, user_job, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat

## 指定用户和电影进行评分
def rating_movie(user_id_val, movie_id_val):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
    
        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference,_, __ = get_tensors(loaded_graph)  #loaded_graph
    
        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]
    
        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]
        
        feed = {
            uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  #x.take(6,1)
            movie_titles: titles,  #x.take(5,1)
            dropout_keep_prob: 1}
    
        # Get Prediction
        inference_val = sess.run([inference], feed)  
    
        return (inference_val)


## 生成movie特征矩阵
def getMovieFec():
    loaded_graph = tf.Graph()  #
    movie_matrics = []
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        user_gender, user_age, user_job, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = get_tensors(loaded_graph)  #loaded_graph

        for item in movies.values:
            categories = np.zeros([1, 18])
            categories[0] = item.take(2)

            titles = np.zeros([1, sentences_size])
            titles[0] = item.take(1)

            feed = {
                movie_categories: categories,  #x.take(6,1)
                movie_titles: titles,  #x.take(5,1)
                dropout_keep_prob: 1}

            movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)  
            movie_matrics.append(movie_combine_layer_flat_val)

    pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))

    
## 生成User特征矩阵
def getUserFec():
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
    loaded_graph = tf.Graph()  #
    users_matrics = []
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        user_gender, user_age, user_job, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, __,user_combine_layer_flat = get_tensors(loaded_graph)  #loaded_graph

        for item in users.values:

            feed = {
                user_gender: np.reshape(item.take(1), [1, 1]),
                user_age: np.reshape(item.take(2), [1, 1]),
                user_job: np.reshape(item.take(3), [1, 1]),
                dropout_keep_prob: 1}

            user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)  
            users_matrics.append(user_combine_layer_flat_val)
    print (len(users_matrics))
    pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))




## 推荐电影
def recommend_same_type_movie(movies5_list, top_k = 3):
    
    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics
        temp=[]
        #推荐同类型的电影
        for movieID in movies5_list:
            probs_embeddings = (movie_matrics[movieid2idx[movieID]]).reshape([1, 200])
            probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
            sim = (probs_similarity.eval())
            p = np.squeeze(sim)
            p[np.argsort(p)[:-top_k]] = 0
            p = p / np.sum(p)
            results = set()
            while len(results) != 1:
                c = np.random.choice(3883, 1, p=p)[0]
                results.add(c)
            for val in (results):
                #print(val)
                #print(movies_orig[val])
                if len(temp)<6:
                    temp.append(movies_orig[val])
                else:
                    break
        return np.array(temp)
        
### 
def recommend_your_favorite_movie(user_id_val,top_k=10):

    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        #推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
    #     print(sim.shape)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)
        
    #     sim_norm = probs_norm_similarity.eval()
    #     print((-sim_norm[0]).argsort()[0:top_k])
    
        #print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        temp=[]
        while len(results) != 6:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            #print(val)
            #print(movies_orig[val])
            if len(temp)<6:
                temp.append(movies_orig[val])

        return np.array(temp)
print (recommend_your_favorite_movie(3,top_k=10))
## 喜欢看这个电影的人还看了哪些电影
import random

def recommend_other_favorite_movie(movie5_list, top_k = 1):

    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        temp=[]
        for movieID in movie5_list:

            probs_movie_embeddings = (movie_matrics[movieID]).reshape([1, 200])
            probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
            favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
     
            probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
            probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
            sim = (probs_similarity.eval())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)
    
    #     print(sim.shape)
    #     print(np.argmax(sim, 1))
            p = np.argmax(sim, 1)
        ######print("喜欢看这个电影的人还喜欢看：")

            results = set()
            while len(results) != 1:
                c = p[random.randrange(top_k)]
                results.add(c)
            for val in (results):
                ######print(val)
                ######print(movies_orig[val])
                if len(temp)<6:
                    temp.append(movies_orig[val])
                else:
                    break        
        return np.array(temp)

    #temp=recommend_other_favorite_movie(1401,20)
    #newUser('F',16,14)
    #title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()