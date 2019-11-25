import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.stats import probplot



def practical_sign(data, feature, label, bins=10):
    """
        Отображение графика практической значимости признака.
        Input params:
            data - датасет (type => pandas.core.frame.DataFrame);
            feature - исследуемый признак (type => str; format columns in data => real number);
            label - метка (type => str; format columns in data => binary|real number);
            bins - количество бинов, на которые разделяется выборка.
        Return:
            None.
    """
    data['bal_bin'], bal_bins = pd.cut(data[feature], bins, retbins=True)
    data_gr = data.groupby("bal_bin")[label].mean().fillna(0.)
    print(data_gr)
    plt.plot(bal_bins[1:], data_gr)
    plt.xlabel(feature)
    plt.ylabel("Proportion of " + label)
    plt.show()

    
def plot_qq(data_feature_false, data_feature_true, feature):
    """
        Функция отображения QQ-графика.
        Input params:
            data_feature_false - дата с меткой 0;
            data_feature_true - дата с меткой 1.
        Return:
            None.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    probplot(data_feature_false[feature], dist="norm", plot=plt)
    plt.title("$Credit Default=0$ probability plot")
    plt.subplot(1, 2, 2)
    probplot(data_feature_true[feature], dist="norm", plot=plt)
    plt.title("$Credit Default=1$ probability plot")
    plt.tight_layout()
    plt.show()    

