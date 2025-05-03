import torch
import numpy as np
torch.manual_seed(0)
## create a vector of shape mxn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print( f'device: {device}')
def print_data( data):
    print('###'* 20)
    if not isinstance(data, int):
        print(f'array: {data}, shape: {data.shape}')
    else:
        print(f'array: {data}')
    print('---'* 20)
m=4
n=3
data = np.random.randint(1,10,size=(1,m))
# create vector of shape 1xm
vector = torch.tensor(data=data, device=device)
print_data( vector )

# create array IN NUMPY of shape mxn
data = np.random.randint(1,10, size=(m,n))
array = torch.tensor(data=data, device=device)
print_data(array)

# reshap array 
reshaped_array = array.reshape((n,m))
print_data(reshaped_array)
reshaped_array = array.reshape((6,2))
print_data(reshaped_array)
reshaped_array = array.reshape((12,-1))
print_data(reshaped_array)
reshaped_array = array.reshape((-1,m*n))
print_data(reshaped_array)

# create  array in pytorch 
array = torch.randint(1,10,size=(m,n), device=device, dtype=torch.float)
print_data(array)

# matrix multiplication
a = torch.randint(1,10,(m,n) )
b = torch.randint(1,10, (n,m))
result = torch.matmul(a,b)
print_data(a)
print_data(b)
print_data(result)

## matrix operations
a_transpose = a.transpose(0,1)
print_data(a_transpose)

b_transpose = b.transpose(0,1)
print_data(b_transpose)

b_1d_matrix = b_transpose.ravel()
print_data(b_1d_matrix)

a_1d_matrix = a_transpose.ravel()
print_data(a_1d_matrix)

a_numel = a_transpose.numel()
print_data( a_numel)