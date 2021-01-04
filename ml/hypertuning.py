from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from time import perf_counter


def find_best_parameters(X_train, y_train, text_column, candidate_params):

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words='english')),
        ("model", LinearSVC(random_state=42, class_weight='balanced'))
    ])

    
    RS = RandomizedSearchCV(pipe, candidate_params, scoring="f1_weighted", n_jobs=4)

    init = perf_counter()
    RS.fit(X_train[text_column], y_train)
    print(f'Total search time: {perf_counter() - init}')

    return RS.best_params_, pipe
