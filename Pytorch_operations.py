import numpy as np

#Converting a Torch Tensor to a NumPy Array
import torch

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

#See how the numpy array changed their value
a.add_(1)
print(a)
print(b)

#Converting Numpy Array to Torch Tensor
#See how changing the np array changed the Torch Tensor automatically

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#Move the tensor to the GPU
r2 = torch.randn(4, 4)
r = torch.rand(4, 4)
add_result = torch.add(r, r2)
r2 = r2.cuda()
print(r2)

#Provide Easy switching between CPU and GPU
CUDA = torch.cuda.is_available()
print(CUDA)
if CUDA:
    add_result = add_result.cuda()
    print(add_result)

#You can also convert a list to a tensor
a = [2, 3, 4, 1]
print(a)
to_list = torch.tensor(a)
print(to_list, to_list.dtype)

data = [[1., 2.], [3., 4.],
        [5., 6.], [7., 8.]]
T = torch.tensor(data)
print(T, T.dtype)

#Tensor Concatenation
first_1 = torch.randn(2, 5)
print(first_1)
second_1 = torch.randn(3, 5)
print(second_1)
#Concatenate along the 0 dimension (concatenate rows)
con_1 = torch.cat([first_1, second_1])
print('\n')
print(con_1)
print('\n')
first_2 = torch.randn(2, 3)
print(first_2)
second_2 = torch.randn(2, 5)
print(second_2)
#COncatenate along the 1 dimension (concatenate columns)
con_2 = torch.cat([first_2, second_2], 1)
print('\n')
print(con_2)
print('\n')

#Adding DIMENSIONS To tensors
tensor_1 = torch.tensor([1, 2, 3, 4])
tensor_a = torch.unsqueeze(tensor_1, 0)
print(tensor_a)
print(tensor_a.shape)
tensor_b = torch.unsqueeze(tensor_1, 1)
print(tensor_b)
print(tensor_b.shape)
print('\n')
tensor_2 = torch.rand(2, 3, 4)
print(tensor_2)
print('\n')
tensor_c = tensor_2[:, :, 2]
print(tensor_c)
print(tensor_c.shape)
print('\n')
tensor_d = torch.unsqueeze(tensor_c, 2)
print(tensor_d)
print(tensor_d.shape)
