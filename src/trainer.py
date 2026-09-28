from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

def training_models(x_train, y_train):
    # One vs ALL (also call one vs rest)
    ova_model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000)
    )
    
    ova_model.fit(x_train, y_train)
    
    # one vs one
    ovo_model = OneVsOneClassifier(
        LogisticRegression(max_iter=1000)
    )
    
    ovo_model.fit(x_train,y_train)
    
    print("\nova and ovo models are trained successfully.")
    
    return ova_model, ovo_model
