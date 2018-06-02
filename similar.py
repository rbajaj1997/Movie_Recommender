import collections
import math
from collections import defaultdict

def calculate_item_similarity(trainset, use_iuf_similarity=False):
    movie_popular, movie_count = calculate_movie_popular(trainset)
    movie_sim_mat = {}
    for user, movies in trainset.items():
        for movie1 in movies:
            movie_sim_mat.setdefault(movie1, defaultdict(int))
            for movie2 in movies:
                if movie1 == movie2:
                    continue
                if use_iuf_similarity:
                    movie_sim_mat[movie1][movie2] += 1 / math.log(1 + len(movies))
                else:
                    movie_sim_mat[movie1][movie2] += 1

    print('calculate item-item similarity matrix...')
    for movie1, related_items in movie_sim_mat.items():
        len_movie1 = movie_popular[movie1]
        for movie2, count in related_items.items():
            len_user2 = movie_popular[movie2]
            movie_sim_mat[movie1][movie2] = count / math.sqrt(len_movie1 * len_user2)
    return movie_sim_mat, movie_popular, movie_count


def calculate_movie_popular(trainset):
    movie_popular = defaultdict(int)
    for user, movies in trainset.items():
        for movie in movies:
            movie_popular[movie] += 1
    movie_count = len(movie_popular)
    return movie_popular, movie_count
