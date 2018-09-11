from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def k_fold_cross_validation(sgd_clf, X_train, y_train_nb):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    for train_index, test_index in skfolds.split(X_train, y_train_nb):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_nb[train_index])
        X_test_folds = X_train[test_index]
        y_test_folds = (y_train_nb[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))
