from src.config import TARGET
from src.data_loader import load_data
from src.preprocessing import preprocessing_data
from src.trainer import training_models
from src.evaluator import evaluate_models
from src.visualizer import plot_confusion_metrices

def main():
    # step1 Load the dataset
    df = load_data()
    
    # step2 retrive the original class names
    class_name = ( df[TARGET].astype("category").cat.categories.to_list() )
    
    #step3 preprocess the data
    x_train, x_test, y_train, y_test, preprocessor = ( preprocessing_data(df) )
    
    #step 4 train both classifier
    ova_model, ovo_model = training_models(
        x_train,
        y_train
    )
    
    
    print("OvA classifiers:", len(ova_model.estimators_))
    print("OvO classifiers:", len(ovo_model.estimators_))
    
    #step5 evaluate both classifiers
    results = evaluate_models(
        ova_model,
        ovo_model,
        x_test,
        y_test
    )
    
    #generate confusion matrix
    plot_confusion_metrices(
        results,
        class_name
    )
    
    #step7 compare accuracy
    print("\nModel Comparision")
    print("-" * 35)
    
    for name, metrics in results.items():
        print(f"{name} Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
            
    
    