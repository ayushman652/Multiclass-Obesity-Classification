
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .config import TARGET

def preprocessing_data(df):
    #seperate input features and target
    x = df.drop(columns=[TARGET])
    y = df[TARGET].astype("category").cat.codes
    
    #identify numericle and categoricle features
    numerical_columns = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = x.select_dtypes(include = ["object", "category"]).columns.tolist()
    
    #split before fitting processing tools
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2, 
        random_state=42,
        stratify=y
        )
    
    #define transformations
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", StandardScaler(), numerical_columns),
            (   "cat",
                OneHotEncoder(
                             drop = "first",
                             handle_unknown = "ignore",
                             sparse_output = False
                             ),
                categorical_columns
            )
        ],
        verbose_feature_names_out = False
    )    
    
    #learn transformaations from training data only
    x_train = preprocessor.fit_transform(x_train)
    
    #apply the same transformation on test data
    x_test = preprocessor.transform(x_test)
    
    print("\nTraining shape: ", x_train.shape)
    print("Testing shape: ", x_test.shape)
    
    return(
        x_train,
        x_test,
        y_train,
        y_test,
        preprocessor
    ) 