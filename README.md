# SVM-PCA-implementation-on-MNIST-dataset

Task 1: Using PCA , reduced the dimensionality of the images from 784 to
 a lower number of principal components (k = {5,10,20,50,100,200,500})

Task 2 : Fitting the MNSIT digits to SVM classifier

Task 3:Perform a comparative study by trying out various combinations of the following:

    • Kernels: use linear, polynomial, and radial basis function kernel.
    
    • C: test out C = {0.1,1,10,100}
    
    • Gamma: (not applicable for linear kernel) test out γ = {1,0.1,0.01,0.001}
    
   For each of the kernels, find out the best combination of hyper-parameters: C and γ (grid search).

Task 4: Evaluate the performance of the model on the test set and report the accuracy, precision,

 recall, and F1 score using the metrics module in Scikit Learn
 

 Task 5: Plot the performance metrics (macro averaged accuracy, precision, recall, and F1 score) with
 respect to the number of principal components used (corresponding to the best hyperparameter
 for that particular reduced PCA embeddings)
