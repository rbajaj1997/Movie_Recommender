import collections
from operator import itemgetter
import math
from collections import defaultdict
import similar
import utils


class ItemBasedCF:
    def __init__(self, k_sim_movie=10, n_rec_movie=5, use_iuf_similarity=False, save_model=True):
        print("ItemBasedCF start...\n")
        self.k_sim_movie = k_sim_movie
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.save_model = save_model
        self.use_iuf_similarity = use_iuf_similarity

    def fit(self, trainset):
        model_manager = utils.ModelManager()
        try:
            self.movie_sim_mat = model_manager.load_model(
                'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat')
            self.movie_popular = model_manager.load_model('movie_popular')
            self.movie_count = model_manager.load_model('movie_count')
            self.trainset = model_manager.load_model('trainset')
            print('Movie similarity model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.movie_sim_mat, self.movie_popular, self.movie_count = \
                similar.calculate_item_similarity(trainset=trainset,
                                                     use_iuf_similarity=self.use_iuf_similarity)
            self.trainset = trainset
            print('Train a new model success.')
            if self.save_model:
                model_manager.save_model(self.movie_sim_mat,
                                         'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat')
                model_manager.save_model(self.movie_popular, 'movie_popular')
                model_manager.save_model(self.movie_count, 'movie_count')
                model_manager.save_model(self.trainset, 'trainset')
                print('The new model has saved success.\n')

    def recommend(self, user):
        K = self.k_sim_movie
        N = self.n_rec_movie
        predict_score = collections.defaultdict(int)
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        watched_movies = self.trainset[user]
        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(), key=itemgetter(1),
                                                           reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                predict_score[related_movie] += similarity_factor * rating
        return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]

    def test(self, testset):
        self.testset = testset
        print('Test recommendation system start...')
        N = self.n_rec_movie
        hit = 0
        rec_count = 0
        test_count = 0

        for i, user in enumerate(self.trainset):
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
            rec_count += N
            test_count += len(test_movies)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        print('Test recommendation system success!')
        print('precision=%.4f\trecall=%.4f\n' % (precision, recall))

    def predict(self, testset):
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        for i, user in enumerate(testset):
            rec_movies = self.recommend(user)  # type:list
            movies_recommend[user].append(rec_movies)
        print('Predict scores success.')
        return movies_recommend

