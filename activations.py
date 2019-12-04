import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    # đạo hàm của hàm tanh ở trên:
    return 1-np.tanh(x)**2;

def softmax(x):
   e = np.exp(x)
   return e / np.sum(e, axis=1)
     
# https://deepnotes.io/softmax-crossentropy
# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
# http://bigstuffgoingon.com/blog/posts/softmax-loss-gradient/
# http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
# https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
# http://saitcelebi.com/tut/output/part2.html

# https://stackoverflow.com/questions/54976533/derivative-of-softmax-function-in-python    
# https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function

# https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d

# Cross Entropy là hàm mất mát - Loss tương ứng với layer softmax:
# https://machinelearningcoban.com/2017/02/17/softmax/
# J(W,xi,yi) = -sum(yji x log(aji) ) với j chạy từ 1 --> C (số class cần phân loại), i chạy từ 1 --> N (số quan sát)
    # xi: dữ liệu đầu vào thứ i
    # yi: giá trị đúng của xi
    # ai = softmax(wT x) trong đó ai là đầu ra dự đoán của điểm dữ liệu đó (T: chuyển vị)
     
# Đạo hàm của hàm mất mát Cross Entropy:
# J(W,xi,yi) = -sum(yji x log(aji) ) với j chạy từ 1 --> C (số class cần phân loại), i chạy từ 1 --> N (số quan sát)
    # xi: dữ liệu đầu vào thứ i
    # yi: giá trị đúng của xi
    # ai = softmax(wT x) trong đó ai là đầu ra dự đoán của điểm dữ liệu đó (T: chuyển vị)
    
# Giả sử rằng chúng ta sử dụng SGD, công thức cập nhật cho ma trận trọng số W sẽ là:    
    # ei = yi - ai
    # ∂J(W)/∂(W) = sum(xi eiT) với i chạy từ 1 -->N (∂J(W)/∂(W): đạo hàm của hàm mất mát Cross Entropy so với W)
    # ∂J(W)/∂(W) = X ET
# W = W + η xi (yi - ai)T, trong đó: η là hệ số learning


def softmax_prime(x):
    # đạo hàm của hàm softmax ở trên:
    # Reshape the 1-d x to 2-d so that np.dot will do the matrix multiplication
    # Vectorized version
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T) # np.diagflat(s): tao 1 ma tran co kich thuoc (x.len, x.len)
    
"""
def softmax_grad(s): 
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])

    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s) #tao 1 ma tran co kich thuoc (s.len, s.len), co gia tri s tren duong cheo chinh

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m
"""    