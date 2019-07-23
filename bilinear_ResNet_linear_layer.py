import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import os
import bilinear_resnet
import CUB_200

base_lr = 0.1
batch_size = 48
num_epochs = 50
weight_decay = 1e-8
num_classes = 200
cub200_path = 'data'
save_model_path = 'model_saved/CUB_200'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    model = bilinear_resnet.BCNN(num_classes, pretrained=True).to(device)
    model_d = model.state_dict()


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    # If the incoming value does not increase for 3 consecutive times, the learning rate will be reduced by 0.1 times
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # Calculate the mean and variance of each channel of sample data, run it only once, and record the corresponding value
    # get_statistic()

    # Mean and variance of CUB_200 dataset are [0.4856, 0.4994, 0.4324], [0.1817, 0.1811, 0.1927]

    # Set up the data preprocessing process
    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                                      torchvision.transforms.CenterCrop(448),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                                       [0.1817, 0.1811, 0.1927])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(448),
                                                     torchvision.transforms.CenterCrop(448),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],
                                                                                      [0.1817, 0.1811, 0.1927])])

    train_data = CUB_200.CUB_200(cub200_path, train=True, transform=train_transform)
    test_data = CUB_200.CUB_200(cub200_path, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    print('Start training the fc layer...')
    best_acc = 0.
    best_epoch = 0
    end_patient = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0.
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

            print('Epoch %d: Iter %d, Loss %g' % (epoch + 1, i + 1, loss))
        train_acc = 100 * correct / total
        print('Testing on test dataset...')
        test_acc = test_accuracy(model, test_loader)
        print('Epoch [{}/{}] Loss: {:.4f} Train_Acc: {:.4f}  Test1_Acc: {:.4f}'
              .format(epoch + 1, num_epochs, epoch_loss, train_acc, test_acc))
        scheduler.step(test_acc)
        if test_acc > best_acc:
            model_file = os.path.join(save_model_path, 'resnet34_CUB_200_train_fc_epoch_%d_acc_%g.pth' %
                                      (best_epoch, best_acc))
            if os.path.isfile(model_file):
                os.remove(os.path.join(save_model_path, 'resnet34_CUB_200_train_fc_epoch_%d_acc_%g.pth' %
                                       (best_epoch, best_acc)))
            end_patient = 0
            best_acc = test_acc
            best_epoch = epoch + 1
            print('The accuracy is improved, save model')
            torch.save(model.state_dict(), os.path.join(save_model_path,
                                                        'resnet34_CUB_200_train_fc_epoch_%d_acc_%g.pth' %
                                                        (best_epoch, best_acc)))
        else:
            end_patient += 1

        # If the accuracy of the 10 iteration is not improved, the training ends
        if end_patient >= 10:
            break
    print('After the training, the end of the epoch %d, the accuracy %g is the highest' % (best_epoch, best_acc))


def test_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
        model.train()
        return 100 * correct / total


def get_statistic():
    train_data = CUB_200.CUB_200(cub200_path, train=True, transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    print('Calculate the mean and variance of the data')
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(mean)
    print(std)


def main():
    train()


if __name__ == '__main__':
    main()
