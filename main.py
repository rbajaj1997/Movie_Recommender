import utils
from ItemCF import ItemBasedCF
from dataset import DataSet

def run_model(model_name, dataset_name, test_size):
    print('*' * 70)
    print('\tThis is %s model trained on %s with test_size = %.2f' % (model_name, dataset_name, test_size))
    print('*' * 70 + '\n')
    model_manager = utils.ModelManager(dataset_name, test_size)
    try:
        trainset = model_manager.load_model('trainset')
        testset = model_manager.load_model('testset')
    except OSError:
        ratings = DataSet.load_dataset(name=dataset_name)
        trainset, testset = DataSet.train_test_split(ratings, test_size=test_size)
        model_manager.save_model(trainset, 'trainset')
        model_manager.save_model(testset, 'testset')
    if model_name == 'ItemCF':
        model = ItemBasedCF()
    else:
        raise ValueError('No model named' + model_name)
    model.fit(trainset)
    recommend_test(model, [1, 55, 233, 666, 888])
    model.test(testset)


def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("Recommend for userid = %s:" % user)
        print(recommend)
        print()


if __name__ == '__main__':
    dataset_name = 'ml-100k'
    # dataset_name = 'ml-1m'
    model_type = 'ItemCF'
    test_size = 0.3
    run_model(model_type, dataset_name, test_size)
