import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self,d):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear_list =[]
        self.d = d
        for i in range(d):
            self.linear_list.append(torch.nn.Linear(d, 1))
        self.linear_list= torch.nn.ModuleList(self.linear_list)



    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.view(len(x),self.d,3)
        a_t = torch.transpose(a,1,2)
        blend = torch.zeros(len(x),3)
        for i in range(self.d):
            pred_1 =self.linear_list[i](a_t[:,0])
            blend =torch.cat((blend,pred_1),1)
        return blend


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

# Create random Tensors to hold inputs and outputs
x = torch.randn(10, 12)
print(x)
# Construct our model by instantiating the class defined above
model = TwoLayerNet(4)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for t in range(1):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    print(y_pred)
    # # Compute and print loss
    # loss = criterion(y_pred, y)
    # if t % 100 == 99:
    #     print(t, loss.item())

    # # Zero gradients, perform a backward pass, and update the weights.
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()