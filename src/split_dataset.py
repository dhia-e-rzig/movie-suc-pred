from sklearn.model_selection import train_test_split
import numpy as np

def split_train_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train = np.delete(x_train,[66,240,267,281,690,818,899,925,955,1101,1217,1503,1553,1777,1786,1800,1848,1891,1958,2036,2109,2324,2372,2438,2506,2607,2796,2834,2843,2878,3009,3155,3784,3190,3219,3224,2357,2369,3326,3476,3479,3553,3644,3713,3732,3791,3795,3945,4019,4118,4222,4287,4325,4385,4410,4460,4486,4500,4504,4556,4642,4655,4841,4895,4896,4965,4974,4976,5004],axis=1)
    x_test = np.delete(x_test,[66,240,267,281,690,818,899,925,955,1101,1217,1503,1553,1777,1786,1800,1848,1891,1958,2036,2109,2324,2372,2438,2506,2607,2796,2834,2843,2878,3009,3155,3784,3190,3219,3224,2357,2369,3326,3476,3479,3553,3644,3713,3732,3791,3795,3945,4019,4118,4222,4287,4325,4385,4410,4460,4486,4500,4504,4556,4642,4655,4841,4895,4896,4965,4974,4976,5004],axis=1)
    return(x_train, x_test, y_train, y_test)