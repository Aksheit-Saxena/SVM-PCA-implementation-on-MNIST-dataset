from utils import get_data, plot_metrics, normalize
from typing import Tuple
import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning



# import the necessary scikit-learn libraries

def main() -> None:
    # hyperparameters
    Cs = [0.1, 1, 10, 100]
    GAMMAs = [1, 0.1, 0.01, 0.001]
    # Run the code 3 times (each time with a different kernel) for comparing the results
    KERNEL = 'linear'  # options: 'linear', 'rbf', 'poly'

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # get validation set (using sklearn train_test_split)
    X_train1,X_val,y_train1,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

    # normalize the data
    X_train1, X_test, X_val = normalize(X_train1, X_val, X_test)

    metric = []
    for k in [5, 10, 20, 50, 100, 200, 500]:

        ##############################
        # YOUR CODE GOES HERE
        ##############################

        # reduce the dimensionality of the data using PCA
        pca = PCA(n_components=k)
        
        X_train2=pca.fit_transform(X_train1)
        X_test2=pca.transform(X_test)
        X_val2=pca.transform(X_val)
        
        # create a model & # train the model with all different combination of hyperparameters (using nested for loops)
        ker=['linear','poly','rbf']
        best_param={key: None for key in ker}
        for ke in ker:
            best_score=-1
            for c in Cs:
                if ke!='linear':
                    for g in GAMMAs:
                        model=svm.SVC(kernel=ke,C=c,gamma=g,max_iter=100)
                        with warnings.catch_warnings():
                          warnings.simplefilter("ignore", category=ConvergenceWarning)
                          model.fit(X_train2,y_train1)
                        score1=model.score(X_val2,y_val)
                        if score1>best_score:
                              best_score=score1
                              best_param[ke]={'C':c,'G':g}
                else:
                    model=svm.SVC(kernel='linear',C=c,max_iter=100)
                    with warnings.catch_warnings():
                      warnings.simplefilter("ignore", category=ConvergenceWarning)
                      model.fit(X_train2, y_train1)
                    score1=model.score(X_val2,y_val)
                    if score1>best_score:
                        best_score=score1
                        best_param['linear']={'C':c} 
        print("\n\nfor k= ",k," : ",best_param)

        # select the best hyperparameter combination and train the model one final time

        # evaluate the model on the test set
        accuracy = 0  # use the accuracy function from SKLearn
        precision = 0  # use the precision function from SKLearn
        recall = 0  # use the recall function from SKLearn
        f1_scor = 0  # use the f1_score function from SKLearn
        
        for m in best_param.keys():
          if m=='linear':
            model=svm.SVC(kernel=m,C=best_param[m]['C'],max_iter=100)
          else:
            model=svm.SVC(kernel=m,C=best_param[m]['C'],gamma=best_param[m]['G'],max_iter=100)
          
          with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train2,y_train1)   
          y_pred=model.predict(X_test2)        
          accuracy = accuracy_score(y_test,y_pred)  # use the accuracy function from SKLearn
          precision =precision_score(y_test,y_pred,average='macro')  # use the precision function from SKLearn
          recall =recall_score(y_test,y_pred,average='macro')  # use the recall function from SKLearn
          f1_scor =f1_score(y_test,y_pred,average='macro')  # use the f1_score function from SKLearn
          metric.append((k, accuracy, precision, recall, f1_scor))
          print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_scor}')

    # plot and save the results
    plot_metrics(metric)


if __name__ == '__main__':
    main()
def main() -> None:
    # hyperparameters
    Cs = [0.1, 1, 10, 100]
    GAMMAs = [1, 0.1, 0.01, 0.001]
    # Run the code 3 times (each time with a different kernel) for comparing the results
    KERNEL = 'linear'  # options: 'linear', 'rbf', 'poly'

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # get validation set (using sklearn train_test_split)
    X_train1,X_val,y_train1,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

    # normalize the data
    X_train1, X_test, X_val = normalize(X_train1, X_val, X_test)

    metric = []
    for k in [5, 10, 20, 50, 100, 200, 500]:

        ##############################
        # YOUR CODE GOES HERE
        ##############################

        # reduce the dimensionality of the data using PCA
        pca = PCA(n_components=k)
        
        X_train2=pca.fit_transform(X_train1)
        X_test2=pca.transform(X_test)
        X_val2=pca.transform(X_val)
        
        # create a model & # train the model with all different combination of hyperparameters (using nested for loops)
        ker=['linear','poly','rbf']
        best_param={key: None for key in ker}
        for ke in ker:
            best_score=-1
            for c in Cs:
                if ke!='linear':
                    for g in GAMMAs:
                        model=svm.SVC(kernel=ke,C=c,gamma=g,max_iter=100)
                        with warnings.catch_warnings():
                          warnings.simplefilter("ignore", category=ConvergenceWarning)
                          model.fit(X_train2,y_train1)
                        score1=model.score(X_val2,y_val)
                        if score1>best_score:
                              best_score=score1
                              best_param[ke]={'C':c,'G':g}
                else:
                    model=svm.SVC(kernel='linear',C=c,max_iter=100)
                    with warnings.catch_warnings():
                      warnings.simplefilter("ignore", category=ConvergenceWarning)
                      model.fit(X_train2, y_train1)
                    score1=model.score(X_val2,y_val)
                    if score1>best_score:
                        best_score=score1
                        best_param['linear']={'C':c} 
        print("\n\nfor k= ",k," : ",best_param)

        # select the best hyperparameter combination and train the model one final time

        # evaluate the model on the test set
        accuracy = 0  # use the accuracy function from SKLearn
        precision = 0  # use the precision function from SKLearn
        recall = 0  # use the recall function from SKLearn
        f1_scor = 0  # use the f1_score function from SKLearn
        
        for m in best_param.keys():
          if m=='linear':
            model=svm.SVC(kernel=m,C=best_param[m]['C'],max_iter=100)
          else:
            model=svm.SVC(kernel=m,C=best_param[m]['C'],gamma=best_param[m]['G'],max_iter=100)
          
          with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train2,y_train1)   
          y_pred=model.predict(X_test2)        
          accuracy = accuracy_score(y_test,y_pred)  # use the accuracy function from SKLearn
          precision =precision_score(y_test,y_pred,average='macro')  # use the precision function from SKLearn
          recall =recall_score(y_test,y_pred,average='macro')  # use the recall function from SKLearn
          f1_scor =f1_score(y_test,y_pred,average='macro')  # use the f1_score function from SKLearn
          metric.append((k, accuracy, precision, recall, f1_scor))
          print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_scor}')

    # plot and save the results
    plot_metrics(metric)


if __name__ == '__main__':
    main()

