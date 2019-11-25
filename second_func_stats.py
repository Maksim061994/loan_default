import numpy as np
import pandas as pd
import math
from sklearn import metrics 
from scipy.stats import norm, chi2_contingency, probplot                                   


def proportions_diff_p_value(z_stat, alternative = "two-sided"):
    """
        Расчёт p-value для множественной проверке гипотезы о равенсте доли в независимых выборках.
        Input params:
            z_stat - z-статистика;
            alternative - тип альтернативы
            (двусторонняя, крайний левый и крайний правый хвост распределения).
        Return:
            p-value.
    """
    if alternative == "two-sided":
        return 2 * (1 - norm.cdf(np.abs(z_stat)))
    
    if alternative == "less":
        return norm.cdf(z_stat)

    if alternative == "greater":
        return 1 - norm.cdf(z_stat)

    
def proportions_diff_z_stat_ind(sample1, sample2):
    """
        Расчёт Z-критерия для разности долей (независимые выборки).
        Input params:
            sample1, sample2 - входные выборки.
        Return:
            z-статистика.
    """
    n1 = len(sample1)
    n2 = len(sample2)
    
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def getVCramer(contingency_matrix):
    """
        Расчёт коэффициента корреляции V-Крамера.
        Input params:
            contingency_matrix - матрица сопряженности.
        Return:
            Коэффициент корреляции.

    """
    chi2 = chi2_contingency(contingency_matrix)[0]
    n = np.ravel(contingency_matrix).sum()
    return math.sqrt(chi2 / (n * (min(contingency_matrix.shape) - 1)))


def count_metrics(model, X_test, y_test, thresholder=0.5):
    """    
        Расчёт метрик.
        Input params:
            model - обученная модель (поддерживается: catboost, sklearn);
            X_test - матрица признаков для тестирования; 
            y_test - вектор меток для тестирования;
            thresholder - порог.
        Return:
            confusion matrix.

    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > thresholder).astype(int)
    columns = ["0", "1"]
    confm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)
    print("Матрица неточностей")
    print("F1 = ", metrics.f1_score(y_test, y_pred))
    print("Recall = ", metrics.recall_score(y_test, y_pred))
    print("Precision = ", metrics.precision_score(y_test, y_pred))
    return df_cm



