import torch
import numpy as np

# Tensors
# Tensors are a specialized data structure that are very similar to arrays and matrices
# In PyTorch, we use tensors to encode the inputs and outputs of a model
# as well as the model's parameters



data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) #retains the propertis of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) #overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#shape is a tuple of tensor dimensions. In the functions below,
#it determines the dimensionality of the output tensor.abs
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor}")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor attributes describe their shape, datatype and the device on which they are stored

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor Operatirs - Over 100 tensor operations
# Each of them can be run on the GPU (typically igher speeds)
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")


tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# You can use torch.cat to concatenate a sequence of tensors along a given dimension
# torch.stack is another joining operation that is subtly different

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#Multiplying tensors
#This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
#Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

#This computes the matrix multiplication between two tensors

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
#Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

#In-place operations
#Operations that have a _ suffix are in-place
#I.E: x.copy_y(), x.t_()
# will change x

print(tensor, "\"n")
tensor.add_(5)
print(tensor)

#Output:
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

# tensor([[6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.]])

# In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

# Tensor to NumPy array
# A change in the tensor reflects in the NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Output:
# t: tensor([2., 2., 2., 2., 2.])
# n: [2. 2. 2. 2. 2.]

# NumPy array to Tensor
# Changes in the NumPy array reflects in the tensor.abs

# torch.autograd is PyTorchs automatic differentiation engine that powers neural network training

# Let’s take a look at a single training step. For this example, 
# we load a pretrained resnet18 model from torchvision.
# We create a random data tensor to represent a single image with 3 channels, 
# and height & width of 64, 
# and its corresponding label initialized to some random values. 
# Label in pretrained models has shape (1,1000).

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64) #1 Image, 3 Channels, Width&Height 64
labels = torch.rand(1, 1000)

# Next we run the input data trough the model trough each of its layers to make a prediction
# This is called the forward pass.

prediction = model(data)

# We use the model's prediction and the corresponding label to calculate the error(loss)
# THe next step is to backpropagate this error trough the network. Backward propagation
# is kicked off when we call .backward() on the error tensor. Autograd then calculates
# and stores the gardients for each model parameter in the parameter's .grad attribute

loss = (prediction - labels).sum()
loss.backward() #backward pass

#Next, we load an optimizer, in this case SGD with a learning rate of 0.01 and momentum of 0.9
# We register all the parameters of the model in the optimizer.

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#Finally we call .step() to initiate gradient descent. The optimizer adjusts each parameter by its gradient
# stored in .grad

optim.step()

# At this point, we have everything you need to train your neural network. The below
# sections detail the workings of autograd - can skip these