import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net = net.to(device)
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open(r'data/plane1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0).to(device) # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].cpu().numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
