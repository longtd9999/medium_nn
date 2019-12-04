# https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65
import numpy as np

# loss function and its derivative
# Tính Error (Loss) của layer (lớp) cuối cùng:
def mse(y_true, y_pred):
    # E = 1 / n * sum( (Y - Y*)^2 )   (n = len(Y) )
    return np.mean(np.power(y_pred - y_true, 2));

# Tính delta_Error/delta_Y của layer (lớp) cuối cùng:
def mse_prime(y_true, y_pred, X):
    #Đạo hàm của hàm mse ở trên:
    # E' = 2 / n * (Y - Y*)   (n = len(Y) )
    return 2 * (y_pred - y_true) / y_true.size;

def cross_entropy(y_true, y_pred):
# cost or loss function là Cross Entropy:
    return -np.sum(y_true * np.log(y_pred))

    
def cross_entropy_prime(y_true, y_pred, X):
    #Tính đạo hàm của hàm Cross Entropy cost(X, Y, W) ở trên:
    # ∂J(W)/∂(W) = X ET = X (Y - A)T (= xi (yi - ai)T với i chạy từ 1 --> N, với N là kích thước dữ liệu đầu vào X)
    
    print("cross_entropy_prime:")
    print("X = ", X, "y_pred=", y_pred, "y_true= ", y_true)    
    
    E = y_pred - y_true
    print("E = ", E)
    print("X.shape = ", X.shape, "E.shape = ", E.shape)
    print("cross_entropy_prime = ", X.dot(E.T))
    return X.dot(E.T)
    
    
"""    
def cross_entropy(X, Y, W):
# cost or loss function là Cross Entropy:
# J(W,xi,yi) = -sum(yji x log(aji) ) với j chạy từ 1 --> C (số class cần phân loại), i chạy từ 1 --> N (số quan sát)
    # xi: dữ liệu đầu vào thứ i
    # yi: giá trị đúng của xi
    # ai = softmax(wT x) trong đó ai là đầu ra dự đoán của điểm dữ liệu đó (T: chuyển vị)
    
    A = softmax(W.T.dot(X))
    return -np.sum(Y * np.log(A))

def cross_entropy_prime(X, Y, W):
    #Tính đạo hàm của hàm Cross Entropy cost(X, Y, W) ở trên:
    # ∂J(W)/∂(W) = X ET = X (Y - A)T (= xi (yi - ai)T với i chạy từ 1 --> N, với N là kích thước dữ liệu đầu vào X)
    
    A = softmax((W.T.dot(X)))
    '''
    array([[0.38683186, 0.12224902],
           [0.39347139, 0.19974992],
           [0.21969674, 0.67800107]])
    '''
    E = A - Y
    return X.dot(E.T)
"""    
    