import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#%matplotlib inline

# import the necessary scikit-learn libraries


# import the necessary scikit-learn libraries


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('mnist_train.csv')
    test_df = pd.read_csv('mnist_test.csv')
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test




def normalize(X_train, X_val, X_test) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # normalize the data
     ##############################
    # YOUR CODE GOES HERE
    ##############################
    
    Scaler=StandardScaler()
    X_train=pd.DataFrame(Scaler.fit_transform(X_train))
    X_test=pd.DataFrame(Scaler.transform(X_test))
    X_val=pd.DataFrame(Scaler.transform(X_val))

    return X_train, X_test, X_val
    raise NotImplementedError

def plot_metrics(metric) -> None:
    # plot and save the results

    ##############################
    # YOUR CODE GOES HERE
    ##############################
    linear_metric=[]
    poly_metric=[]
    rbf_metric=[]
    for t in range(len(metric)):
        z=t%3
        if z==0:
          linear_metric.append(metric[t][1:])
        elif z==1:
          poly_metric.append(metric[t][1:])
        else:
          rbf_metric.append(metric[t][1:])
    print('##################################################################################################################')
    print("#################  Graphs depicting metrics v/s k values for kernels- linear, ploynomial & rbf  #############")
    plt.grid()
    plt.xlabel("Number of Principal Component (k)")
    plt.ylabel("Metric Scores")
    plt.legend(loc='best',title='Linear kernel : Metrics v/s k')
    plt.plot([5, 10, 20, 50, 100, 200, 500], linear_metric,marker='*',label="Linear Kernel")
    plt.text(7, linear_metric[0][0], "Accuracy")
    plt.text(12,linear_metric[1][1], "Precision")
    plt.text(22, linear_metric[2][2], "Recall")
    plt.text(22, linear_metric[3][3], "F1 Score")
    plt.show()
    plt.savefig("Linear")


    plt.grid()
    plt.xlabel("Number of Principal Component (k)")
    plt.ylabel("Metric Scores")
    plt.legend(loc='best',title='Polynomial kernel : Metrics v/s k')
    plt.plot([5, 10, 20, 50, 100, 200, 500], poly_metric,marker='^',label="Polynomial Kernel")
    plt.text(7, poly_metric[0][0]+0.2, "Accuracy")
    plt.text(12,poly_metric[1][1]+0.3, "Precision")
    plt.text(22, poly_metric[2][2]+0.2, "Recall")
    plt.text(22, poly_metric[3][3]+0.1, "F1 Score")
    plt.show()
    plt.savefig("Polynomial")

    plt.grid()
    plt.xlabel("Number of Principal Component (k)")
    plt.ylabel("Metric Scores")
    plt.legend(loc='best',title='Radial Basis Function kernel : Metrics v/s k')
    plt.plot([5, 10, 20, 50, 100, 200, 500],rbf_metric,marker='o',label="RBF Kernel")
    plt.text(7, rbf_metric[0][0], "Accuracy")
    plt.text(12,rbf_metric[1][1], "Precision")
    plt.text(22, rbf_metric[2][2], "Recall")
    plt.text(22, rbf_metric[3][3], "F1 Score")
    plt.show()
    plt.savefig("Radial Basis Function")

    print('##################################################################################################################')
    
    raise NotImplementedError
