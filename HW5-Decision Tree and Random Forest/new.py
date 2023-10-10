import pandas as pd
from decision_tree_starter import *


path_train = './dataset/titanic/titanic_training.csv'
data = genfromtxt(path_train, delimiter=',', dtype=None, encoding=None)
path_test = './dataset/titanic/titanic_test_data.csv'
test_data = genfromtxt(path_test, delimiter=',', dtype=None, encoding=None)
y = data[1:, -1]  # label = survived
class_names = ["Died", "Survived"]
labeled_idx = np.where(y != '')[0]

y = np.array(y[labeled_idx])
y = y.astype(float).astype(int)

X, onehot_features = preprocess(data[1:, :-1], onehot_cols=[1, 5, 7, 8])
X = X[labeled_idx, :]
Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
assert X.shape[1] == Z.shape[1]
features = list(data[0, :-1]) + onehot_features


tree = DecisionTree(feature_labels=features)
tree.fit(X,y)



spam_features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]


# # Load spam data
# path_train = './dataset/spam/spam_data.mat'
# data = scipy.io.loadmat(path_train)
# s_X = data['training_data']
# s_y = np.squeeze(data['training_labels'])
# s_Z = data['test_data']
# class_names = ["Ham", "Spam"]
# spam_tree = BaggedTrees(feature_labels=spam_features)
# spam_tree.fit(s_X,s_y)
# print(np.sum(spam_tree.predict(s_X)==s_y)/len(s_y))
# spam_result = spam_tree.predict(s_Z).astype(int)
# spam_df = pd.DataFrame(spam_result)
# spam_df.index+=1
# spam_df.to_csv('spam.csv',index_label='index')