import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

class MNIST_PREDICTION():
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.all_probas = {}
            self.model_names = []
        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


# ============= TRAIN PERFORMANCE ================ #
    def train_performance(self, model):
        try:
            print("Train Accuracy :", accuracy_score(self.y_train, model.predict(self.X_train)))
            print("Train Confusion :")
            print(confusion_matrix(self.y_train, model.predict(self.X_train)))
            print("Train Report :")
            print(classification_report(self.y_train, model.predict(self.X_train)))
        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


 # ============= TEST PERFORMANCE ================ #
    def test_performance(self, model):
        try:
            print("Test Accuracy :", accuracy_score(self.y_test, model.predict(self.X_test)))
            print("Test Confusion :")
            print(confusion_matrix(self.y_test, model.predict(self.X_test)))
            print("Test Report :")
            print(classification_report(self.y_test, model.predict(self.X_test)))

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    # ===================== MODELS ======================= #
    def knn(self):
        try:
            print("\n===== KNN =====")
            self.knn_model = KNeighborsClassifier(n_neighbors=5)
            self.knn_model.fit(self.X_train, self.y_train)
            self.all_probas["KNN"] = self.knn_model.predict_proba(self.X_test)
            self.model_names.append("KNN")
            self.train_performance(self.knn_model)
            self.test_performance(self.knn_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def nb(self):
        try:
            print("\n===== NB =====")
            self.nb_model = GaussianNB()
            self.nb_model.fit(self.X_train, self.y_train)
            self.all_probas["NB"] = self.nb_model.predict_proba(self.X_test)
            self.model_names.append("NB")
            self.train_performance(self.nb_model)
            self.test_performance(self.nb_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def lr(self):
        try:
            print("\n===== Logistic Regression =====")
            self.lr_model = LogisticRegression()
            self.lr_model.fit(self.X_train, self.y_train)
            self.all_probas["LR"] = self.lr_model.predict_proba(self.X_test)
            self.model_names.append("LR")
            self.train_performance(self.lr_model)
            self.test_performance(self.lr_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def dt(self):
        try:
            print("\n===== Decision Tree =====")
            self.dt_model = DecisionTreeClassifier(criterion='entropy')
            self.dt_model.fit(self.X_train, self.y_train)
            self.all_probas["DT"] = self.dt_model.predict_proba(self.X_test)
            self.model_names.append("DT")
            self.train_performance(self.dt_model)
            self.test_performance(self.dt_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def rf(self):
        try:
            print("\n===== Random Forest =====")
            self.rf_model = RandomForestClassifier(n_estimators=20, criterion='entropy')
            self.rf_model.fit(self.X_train, self.y_train)
            self.all_probas["RF"] = self.rf_model.predict_proba(self.X_test)
            self.model_names.append("RF")
            self.train_performance(self.rf_model)
            self.test_performance(self.rf_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def ada(self):
        try:
            print("\n===== AdaBoost =====")
            t = LogisticRegression()
            self.ada_model = AdaBoostClassifier(estimator=t, n_estimators=5)
            self.ada_model.fit(self.X_train, self.y_train)
            self.all_probas["ADABOOST"] = self.ada_model.predict_proba(self.X_test)
            self.model_names.append("ADABOOST")
            self.train_performance(self.ada_model)
            self.test_performance(self.ada_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def gb(self):
        try:
            print("\n===== Gradient Boosting =====")
            self.gb_model = GradientBoostingClassifier(n_estimators=10)
            self.gb_model.fit(self.X_train, self.y_train)
            self.all_probas["GB"] = self.gb_model.predict_proba(self.X_test)
            self.model_names.append("GB")
            self.train_performance(self.gb_model)
            self.test_performance(self.gb_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def xgb_(self):
        try:
            print("\n===== XGBoost =====")
            self.xgb_model = XGBClassifier()
            self.xgb_model.fit(self.X_train, self.y_train)
            self.all_probas["XGB"] = self.xgb_model.predict_proba(self.X_test)
            self.model_names.append("XGB")
            self.train_performance(self.xgb_model)
            self.test_performance(self.xgb_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    def svm_c(self):
        try:
            print("\n===== SVM =====")
            self.svm_model = SVC(kernel='rbf', probability=True)
            self.svm_model.fit(self.X_train, self.y_train)
            self.all_probas["SVM"] = self.svm_model.predict_proba(self.X_test)
            self.model_names.append("SVM")
            self.train_performance(self.svm_model)
            self.test_performance(self.svm_model)

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    # ===================== ALL MODELS ============================#
    def common(self):
        try:
            self.knn()
            self.nb()
            self.lr()
            self.dt()
            self.rf()
            self.ada()
            self.gb()
            self.xgb_()
            self.svm_c()
        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    # ===================== AUC-ROC ========================== #
    def auc_roc(self):
        try:
            y_binary = label_binarize(self.y_test, classes=list(range(10)))
            self.roc_data = {}
            for name in self.model_names:
                prob = self.all_probas[name]
                auc_val = roc_auc_score(y_binary, prob, average='macro')
                fpr = {}
                tpr = {}
                for i in range(y_binary.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], prob[:, i])
                self.roc_data[name] = (fpr, tpr, auc_val)
                print(f"{name} AUC: {auc_val}")
            best = None
            best_auc = -1
            for name in self.roc_data:
                if self.roc_data[name][2] > best_auc:
                    best_auc = self.roc_data[name][2]
                    best = name

            print("\n===== BEST MODEL =====")
            print(f"{best} with AUC = {best_auc}")

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    # ===================== PLOT AUC-ROC ========================== #
    def plot(self):
        try:
            plt.figure(figsize=(8, 6))
            for name, (fpr, tpr, auc_val) in self.roc_data.items():
                mean_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.mean([np.interp(mean_fpr, fpr[i], tpr[i]) for i in fpr], axis=0)
                mean_tpr[0] = 0.0
                plt.plot(mean_fpr, mean_tpr, label=f"{name}")

            plt.plot([0, 1], [0, 1], "k--")
            plt.title("AUC-ROC Curve - All Models")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.show()

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")


    # ===================== HYPERPARAMETER TUNING ========================== #
    def hyperparameter_tuning(self):
        try:
            self.svm_model = SVC(kernel='rbf', probability=True)
            self.parameters_from_svm = {
                'C':[1.0,2.0,3.0],
                'kernel':['linear','poly','rbf','sigmoid'],
                'class_weight':[None,'balanced']
            }
            self.grid_model = GridSearchCV(
                estimator=self.svm_model,
                param_grid=self.parameters_from_svm,
                cv=10,
                scoring='accuracy'
            )
            result = self.grid_model.fit(self.X_train, self.y_train)
            print("Best Parameters:", self.grid_model.best_params_)
            print("Best Score:", self.grid_model.best_score_)
            return self.grid_model.best_params_

        except Exception as e:
            et, em, el = sys.exc_info()
            print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")

# ================ MAIN ========================= #

if __name__ == '__main__':
    try:
        dataset_path = 'mnist_train.csv'
        obj = MNIST_PREDICTION(dataset_path)
        obj.common()
        obj.auc_roc()
        obj.plot()
        obj.hyperparameter_tuning()

    except Exception as e:
        et, em, el = sys.exc_info()
        print(f"Error Type:{et}  Message:{em}  Line:{el.tb_lineno}")

