import pandas as pd
import numpy as np
from numpy import exp
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split


def split_data(x, y):
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.8, random_state=42
    )
    splitted = {
        "train": {"x": x_train, "y": y_train},
        "validation": {"x": x_validation, "y": y_validation},
    }

    return splitted


def get_loss_function(y):
    if len(set(y)) == 2:
        return "Logloss"
    else:
        return "MultiClass"


def CatBoostClassifierModel(train_df, target_name, cat_features):

    y = train_df[target_name]
    x = train_df.drop(target_name, axis=1)
    pool = Pool(data=x, label=y, cat_features=cat_features)
    data = split_data(x, y)

    print(f"y-values: {set(y)}")

    loss_function = get_loss_function(y)
    # also check https://catboost.ai/docs/features/overfitting-detector-desc.html for overfitting handling
    model = CatBoostClassifier(
        iterations=10,
        learning_rate=1,
        loss_function=loss_function,
        # custom_loss=["AUC", "Accuracy"],
    )

    model.fit(
        data["train"]["x"],
        data["train"]["y"],
        cat_features=cat_features,
        eval_set=(data["validate"]["x"], data["validate"]["y"]),
        plot=True,
    )

    # Cross-Validation

    # Will give overview of values for all iterations
    cv_info = cv(
        params=model.get_params(),
        pool=pool,
        fold_count=5,
        shuffle=True,
        partition_random_seed=0,
        plot=True,
        stratified=False,  # or True
        verbose=False,
    )

    # ### if minimizing col
    # print(cv_info.head())
    # column_to_optimize = input("What is the column you want to optimize")
    # column_to_optimize = "fill in value"
    # best_value = np.min(cv_info[column_to_optimize])
    # best_iteration = np.argmin(cv_info[column_to_optimize])
    # print(f'Best value for {column_to_optimize} is {best_value} on iteration{best_iteration}')

    # predict
    preds = model.predict(data=data["validation"]["x"])
    preds_proba = model.predict_proba(data=data["validation"]["x"])

    sigmoid = lambda x: 1 / (1 + exp(-x))
    probs = sigmoid(
        model.predict(data=data["validation"]["x"], prediction_type="RawFormulaVal")
    )

    print(preds)
    print(preds_proba)
    print(probs)


# interesting example found https://github.com/catboost/tutorials/blob/master/classification/classification_tutorial.ipynb

