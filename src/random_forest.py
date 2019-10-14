from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from split_dataset import split_train_test
import numpy as np
import matplotlib.pyplot as plt


def run_random_forest(df_knn):
    print("\n\n----------------------Random Forest----------------------\n\n")

    y = np.array(df_knn["class"])
    x = np.array(df_knn.drop(columns="class"))

    x_train, x_test, y_train, y_test = split_train_test(x, y)

    rf = RandomForestClassifier(
        random_state=1, n_estimators=250, min_samples_split=8, min_samples_leaf=4
    )
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    test_pred = rf.predict(x_train)
    
    success_indices = np.where(y_train==2)[0]
    test_frame = x_train[success_indices]
    print("location index:", success_indices)
    tst_y = y_train[success_indices]
    pred_frame = rf.predict(test_frame)
    print(
        "\n\n----------------------Pred success----------------------\n\n",
        accuracy_score(tst_y, pred_frame)
    )
    

    failure_indices = np.where(y_train==1)[0]
    test_frame = x_train[failure_indices]
    tst_y = y_train[failure_indices]
    pred_frame = rf.predict(test_frame)
    print(
        "\n\n----------------------Pred Failure----------------------\n\n",
        accuracy_score(tst_y, pred_frame)
    )
    
    good_indices = np.where(y_train==1)[0]
    test_frame = x_train[good_indices]
    tst_y = y_train[good_indices]
    pred_frame = rf.predict(test_frame)
    print(
        "\n\n----------------------Pred good----------------------\n\n",
        accuracy_score(tst_y, pred_frame)
    )
    
    # print the Training Accuracy
    print(
        "\n\n----------------------Training Set Accuracy----------------------\n\n",
        accuracy_score(y_train, test_pred),
    )
    print(
        "\n\n----------------------Testing Set Accuracy----------------------\n\n",
        accuracy_score(y_test, pred),
    )

    return rf
