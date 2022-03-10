import torch
#This is a 1-D Tensor
a = torch.tensor([2, 2, 1])
print(a)

#This is a 2-D Tensor
b = torch.tensor([[2, 1, 4],[3, 5, 4], [1, 2, 0], [4, 3, 2]])
print(b)

#The size of the tensors
print(a.shape)
print(b.shape)
print(a.size())
print(b.size())

#Get the height/number of rows of b
print(b.shape[0])

c = torch.FloatTensor([[2, 1, 4],[3, 5, 4], [1, 2, 0], [4, 3, 2]])
#Or we can do
#c = torch.tensor([2, 2, 1], dtype = torch.float)

d = torch.DoubleTensor([[2, 1, 4],[3, 5, 4], [1, 2, 0], [4, 3, 2]])
#or we can do
#d = torch.tensor([2, 2, 1], dtype = torch.double)

print(c)
print(c.dtype)

print(d)
print(d.dtype)

print(c.mean())
print(d.mean())

print(c.std())
print(d.std())

#Reshape b
#Note; If one of the dimensions is -1, its size can be inferred
print(b.view(-1, 1))
print(b.view(12))
print(b.view(-1, 4))
print(b.view(3, 4))
#Assign b a new shape
b = b.view(1, -1)
print(b)
print(b.shape)
#We can even reshape 3D tensors
print('\n')
#Create a 3D Tensor with 2 channels, 3 rows and 4 columns (channels, rows, columns)
three_dim = torch.randn(2, 3, 4)
print('\n')
print(three_dim)
print(three_dim.view(2, 12)) #Reshape to 2 rows, 12 columns
print(three_dim.view(2, -1))

#Create a matrix with random numbers between 0 and 1
r = torch.rand(4, 4)
print(r)

#Create a matrix with random numbers taken from a normal, distribution with a mean 0 and variance 1
r2 = torch.randn(4, 4)
print(r2)
print(r2.dtype)

#Create an array of 5 random integers from values between 6 and 9 (exclusive of 10)
in_array = torch.randint(6, 10, (5,))
print(in_array)
print(in_array.dtype)

#Create a 2-D array (or matrix) of size 3x3 filled with random integers from values between 6 and 9 (exclusive of 10)
in_array2 = torch.randint(6, 10, (3, 3))
print(in_array2)

#Get the number of elements in in_array
print(torch.numel(in_array))
#Get the number of elements in in_array
print(torch.numel(in_array2))

#COnstruct a 3x3 matrix of zeros and of dtype Long:
z = torch.zeros(3, 3, dtype=torch.long)
print(z)
#Construct a 3x3 matrix of tones
o = torch.ones(3, 3)
print(o)
print(o.dtype)

r2_like = torch.randn_like(r2, dtype=torch.double) #COnvert the data tyoe of the tensor
print(r2_like)

#Add two tensors, make sure they are the same size and data type
add_result = torch.add(r, r2)
print(add_result)

#In-place addition (change the value of r2)
r2.add_(r)
print(r2)

print(r2[:, 1])
print(r2[:, :2])
print(r2[:3, :])
num_ten = r2[2, 3]
print(num_ten)
print(num_ten.item())
print(r2[2, :])
