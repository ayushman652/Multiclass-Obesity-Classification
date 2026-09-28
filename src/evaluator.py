from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_models(ova_model, ovo_model, x_test, y_test):
    results = {}
    
    models = {
        "OvA" : ova_model,
        "OvO" : ovo_model
    }
    
    for name , model in models.items():
        y_pred = model.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'=' * 40}")
        print(f"{name} Evaluation")
        print(f"{'=' * 40}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(report)
        print("\nConfusion matrix")
        print(cm)
        
        results[name] ={
            "accuracy" : accuracy,
            "predictions" : y_pred,
            "confusion_matrix" : cm
        } 
        
    return results