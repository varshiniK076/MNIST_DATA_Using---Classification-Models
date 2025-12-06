import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

class BEST_SVM:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.y_test_bin = label_binarize(self.y_test, classes=list(range(10)))

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Error Type : {ex_type}, Error Message : {ex_msg}, Line : {ex_line.tb_lineno}")


# ================= SVM(BEFORE TUNING) =================  #
    def old_svm_run(self):
        try:
            print("\n===== SVM (Before Tuning) =====")

            self.old_svm = SVC(kernel='rbf', probability=True)
            self.old_svm.fit(self.X_train, self.y_train)
            print("Train Accuracy:", accuracy_score(self.y_train, self.old_svm.predict(self.X_train)))
            print("Test Accuracy :", accuracy_score(self.y_test, self.old_svm.predict(self.X_test)))
            print("\nTest Report:\n", classification_report(self.y_test, self.old_svm.predict(self.X_test)))
            old_proba = self.old_svm.predict_proba(self.X_test)
            self.fpr_old, self.tpr_old, _ = roc_curve(self.y_test_bin[:, 0], old_proba[:, 0])
            self.old_auc = auc(self.fpr_old, self.tpr_old)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Error Type : {ex_type}, Error Message : {ex_msg}, Line : {ex_line.tb_lineno}")


# ================= SVM(AFTER TUNING) =================  #
    def best_svm_run(self):
        try:
            print("\n===== SVM (After Tuning) =====")
            self.best_svm = SVC(C=3.0, kernel='rbf', class_weight=None, probability=True)
            self.best_svm.fit(self.X_train, self.y_train)
            print("Train Accuracy:", accuracy_score(self.y_train, self.best_svm.predict(self.X_train)))
            print("Test Accuracy :", accuracy_score(self.y_test, self.best_svm.predict(self.X_test)))
            print("\nTest Report:\n", classification_report(self.y_test, self.best_svm.predict(self.X_test)))
            best_proba = self.best_svm.predict_proba(self.X_test)
            self.fpr_best, self.tpr_best, _ = roc_curve(self.y_test_bin[:, 0], best_proba[:, 0])
            self.best_auc = auc(self.fpr_best, self.tpr_best)
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Error Type : {ex_type}, Error Message : {ex_msg}, Line : {ex_line.tb_lineno}")


# ================= AUC-ROC CURVE =================  #
    def plot(self):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(self.fpr_old, self.tpr_old, label=f"Old SVM (AUC = {self.old_auc:.3f})")
            plt.plot(self.fpr_best, self.tpr_best, label=f"Best SVM (AUC = {self.best_auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("ROC Curve: Old SVM vs Best SVM")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.show()
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Error Type : {ex_type}, Error Message : {ex_msg}, Line : {ex_line.tb_lineno}")

# ================= MAIN ================= #

if __name__ == "__main__":
    try:
        obj1 = BEST_SVM("mnist_train.csv")
        obj1.old_svm_run()
        obj1.best_svm_run()
        obj1.plot()
    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        print(f"Error Type : {ex_type}, Error Message : {ex_msg}, Line : {ex_line.tb_lineno}")
