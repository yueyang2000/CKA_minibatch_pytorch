from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.models import resnet18
import torchvision.transforms as transforms

from cka import CKA_Minibatch_Grid


def forward_features(model, x):
    _b = x.shape[0]
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x1 = x
    x = model.layer2(x)
    x2 = x
    x = model.layer3(x)
    x3 = x
    x = model.layer4(x)
    x4 = x
    return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1)



def main():
    DATA_ROOT = '/home/data/ImageNet/val'
    batch_size = 128
    dataset_size = 1280
    num_sweep = 10
    num_features = 4

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    torch.random.manual_seed(0)
    perms = [torch.randperm(dataset_size) for _ in range(num_sweep)]
    dataset = datasets.ImageFolder(DATA_ROOT, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    model = resnet18(pretrained=True)
    model.cuda()
    model.eval()
    cka_logger = CKA_Minibatch_Grid(num_features, num_features)
    with torch.no_grad():
        for sweep in range(num_sweep):
            dataset_sweep = torch.utils.data.Subset(dataset, perms[sweep])
            data_loader = torch.utils.data.DataLoader(
                dataset_sweep,
                batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True)
            for images, targets in tqdm(data_loader):
                images = images.cuda()
                features = forward_features(model, images)
                cka_logger.update(features, features)
                torch.cuda.empty_cache()

    cka_matrix = cka_logger.compute()

    plt.title('Pretrained Resnet18 Layer CKA')
    plt.xticks([0, 1, 2, 3], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])
    plt.yticks([0, 1, 2, 3], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])
    plt.imshow(cka_matrix.numpy(), origin='lower', cmap='magma')
    plt.clim(0, 1)
    plt.colorbar()
    plt.savefig('r18_cka_new.png')

if __name__ == '__main__':
    main()