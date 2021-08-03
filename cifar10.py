import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms



class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(train_dataloader, model, loss_func, optimizer):
    size = len(train_dataloader.dataset)
    running_loss = 0
    for batch, (x,y) in enumerate(train_dataloader, 0):
        pred = model(x.to(device))
        loss = loss_func(pred, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 2000 == 0:
            print("[%5d] loss: %0.3f"%(batch+1, running_loss/2000))
            running_loss = 0

def test_loop(test_dataloader, model, loss_func):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    total, correct = 0, 0
    with torch.no_grad():
        for x,y in test_dataloader:
            output = model(x.to(device))
            _, pred = torch.max(output.data,1)
            total += y.size(0)
            # total_loss += loss_func(pred,y)
            correct+=(pred == y.to(device)).sum().item()

    # test_loss = total_loss/size
    # correct /= size
    print(f"Accuracy: {(100*correct/total):>0.1f}% \n")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_datasets = datasets.CIFAR10(root=r'E:\Dataset\Cifar10', download=True, train=True, transform=transform)
    test_datasets = datasets.CIFAR10(root=r'E:\Dataset\Cifar10', download=True, train=False, transform=transform)

    train_dataloader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=64, shuffle=True)

    model = NeuralNet().to(device)
    epoch = 20
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for i in range(epoch):
        print("The epoch is: {}--------------------------".format(i+1))
        train_loop(train_dataloader, model, loss_func, optimizer)
        test_loop(test_dataloader, model, loss_func)

    torch.save(model.state_dict(), "cifar10_weight.pth")
    print("Down!")
