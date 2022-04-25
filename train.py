import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #     # 50000张训练图片
    #     # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=4,
                                               pin_memory=True, drop_last=True)
    #     # 10000张验证图片
    #     # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128,
                                             shuffle=False, num_workers=4,
                                             pin_memory=True, drop_last=True)
    # val_data_iter = iter(val_loader)
    # val_image, val_label = val_data_iter.next()
    # val_image, val_label = torch.tensor(val_image), torch.tensor(val_label)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total = 0  # 测试集相片总数
            correct = 0  # 预测对的相片总数
            if step % 100 == 99:  # print every 500 mini-batches
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        outputs = net(val_images)  # [batch, 10]
                        _, predict_y = torch.max(outputs, dim=1)
                        correct += torch.eq(predict_y, val_labels).sum().item()
                        total += val_labels.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % (epoch + 1, step + 1,
                                                                               running_loss / 100,
                                                                               100 * correct / total))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)
# branch
