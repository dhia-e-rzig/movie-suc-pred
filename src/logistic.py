from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import numpy as np
from split_dataset import split_train_test

def run_logistic_regression(df_knn):

    y = np.array(df_knn["class"])
    x = np.array(df_knn.drop(columns="class"))

    x_train, x_test, y_train, y_test = split_train_test(x, y)


    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    
    pred_train = logistic.predict(x_train)
    print ("Training accuracy: ", (logistic.score(x_train, y_train) * 100))

    pred_test = logistic.predict(x_test)
    print ("Testing accuracy: ", (logistic.score(x_test, y_test) * 100))

    success_indices = np.where(y_train==2)[0]
    test_frame = x_train[success_indices]
    tst_y = y_train[success_indices]
    pred_frame = logistic.predict(test_frame)
    print(
        "\n\n----------------------Pred success----------------------\n\n",
        logistic.score(test_frame, tst_y)
    )
    

    failure_indices = np.where(y_train==1)[0]
    test_frame = x_train[failure_indices]
    tst_y = y_train[failure_indices]
    pred_frame = logistic.predict(test_frame)
    print(
        "\n\n----------------------Pred Failure----------------------\n\n",
        logistic.score(test_frame, tst_y)
    )
    
    good_indices = np.where(y_train==1)[0]
    test_frame = x_train[good_indices]
    tst_y = y_train[good_indices]
    pred_frame = logistic.predict(test_frame)
    print(
        "\n\n----------------------Pred good----------------------\n\n",
        logistic.score(test_frame, tst_y)
    )
    return logistic

