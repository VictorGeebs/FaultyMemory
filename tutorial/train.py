import torch
import torch.nn as nn
import torch.optim as optim
import FaultyMemory.utils as utils

PATH = "./models/xor_net.pth"


trainset = utils.R2Dataset(2, 1000)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=5, shuffle=True, num_workers=2
)

testset = utils.R2Dataset(2, 10000)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

# Loading Net
net = utils.Xor()
net.load_state_dict(torch.load(PATH))

# Initial Test
init_accuracy = utils.test_accuracy(net, testloader)
print("Initial accuracy of the network: %3.2f %%" % (100 * init_accuracy))

# Defining loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

utils.train_net(net, optimizer, criterion, trainloader, nb_epochs=100, prt=False)

# Final test
accuracy = utils.test_accuracy(net, testloader)
print("Accuracy of the network: %3.2f %%" % (100 * accuracy))

# Saving net
if accuracy > init_accuracy:
    print("saving net")
    torch.save(net.state_dict(), PATH)
