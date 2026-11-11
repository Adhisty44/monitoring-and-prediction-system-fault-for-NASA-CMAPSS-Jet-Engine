from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from training.base import BaseTrainer, RANDOM_STATE

class SVMTrainer(BaseTrainer):
    @property
    def name(self):
        return "svm"

    def build_model(self):
        return LinearSVC(class_weight="balanced", max_iter=20000,
                         random_state=RANDOM_STATE)



class RandomForestTrainer(BaseTrainer):
    @property
    def name(self):
        return "random_forest"

    def build_model(self):
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=-1, random_state=RANDOM_STATE)


TRAINERS = {
    "svm": SVMTrainer,
    "random_forest": RandomForestTrainer,
}

