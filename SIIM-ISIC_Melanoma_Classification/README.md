# How to deal with the imbalance dataset
1. Collect more dataset, especially the minority one.  
2. 
## Metrics
1. Confusion matrix
2. Precision: how trustable is the result when the model answer that a point belongs to that class
3. Recall: how well the model is able to detect that class.
4. F1 score: (2*precision*recall) / (precision+recall)
5. high recall + high precision: the class is perfectly handled by the model
6. low recall + high precision : the model can’t detect the class well but is highly trustable when it does  
7. high recall + low precision : the class is well detected but the model also include points of other classes in it  
8. low recall + low precision : the class is poorly handled by the model
9. ROC curve: 
10. AUC: