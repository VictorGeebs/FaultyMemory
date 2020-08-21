import torch
import torchvision
import torchvision.transforms as transforms
import FaultyMemory.handler as H
import FaultyMemory.utils as utils
import wrn_mcdonnell_manual as McDo
import Dropit
from collections import OrderedDict

PATH = './models/mcdonnell.pth'

# all weights binary

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)


# init params: depth=28, width=10
net = McDo.WRN_McDonnell(depth=28, width=10, num_classes=10, dropit=False, actprec=3)
state_dict = torch.load(PATH, map_location=torch.device('cpu'))['model_state_dict']
lanmax_state_space = torch.load(PATH, map_location=torch.device('cpu'))['lanmax_state_space']

net.load_state_dict(state_dict)


handler = H.Handler(net)
handler.from_json('./profiles/McDo.json')


# pert_index = 0
# for name in dict(net.named_parameters()):
#     info = handler.tensor_info[name]
#     if info[1] is not None:
#         pert = info[1][0]
#         #pert.set_probability(0.5)
#         pert.set_probability(lanmax_state_space[pert_index])
#     pert_index += 1

# print("perturbing")
# handler.perturb_tensors(scaling=True)
# print("perturbed")
# handler.to_json('./profiles/McDo.json')

# print("starting testing")
# start_time = time.time()

# results = utils.test_accuracy(net, testloader)
# tot_time = time.time()-start_time
# print("Acc: ", results)
# print("Time: ", tot_time)

acc_dict = {}
for scaling in [True, False]:
    nb_nets = 5
    avg_list = []
    for net_index in range(nb_nets):
        print("perturbing net")
        handler.perturb_tensors(scaling)
        avg_list.append(utils.test_accuracy(net, testloader))
        handler.restore()
    acc_dict[scaling] = avg_list

print("Accuracies", acc_dict)


"""
probs = np.logspace(-0.1, -2.5, 20)
print(probs)

clean_accuracy, pert_accuracy, acti_accuracy, both_accuracy = utils.generate_graphs(net, testloader, probs)

plt.figure(1)
plt.plot(probs, clean_accuracy, probs, pert_accuracy, probs, acti_accuracy, probs, both_accuracy)
plt.title("Accuracy with regard to probability")
plt.xlabel("probability")
plt.ylabel("accuracy")
plt.legend(["clean", "only weights", "only activations", "weights and actis"])
plt.show()
"""