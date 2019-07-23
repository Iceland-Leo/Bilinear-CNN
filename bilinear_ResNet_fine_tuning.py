import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import os
import bilinear_resnet
import CUB_200

base_lr = 0.001
batch_size = 24
num_epochs = 50
weight_decay = 1e-5
num_classes = 200
cub200_path = '../../dataset/CUB_200_2011_data'
save_model_path = 'model_saved/CUB_200'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    model = bilinear_resnet.BCNN(num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(save_model_path,
                                                  'resnet34_CUB_200_fine_tuning_epoch_30_acc_83.8281.pth'),
                                                  map_location=lambda storage, loc: storage))
    model_d = model.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    # 若传入值连续3次不增加，则0.1的步伐降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # 设置数据的预处理过程
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

    print('开始微调整个网络...')
    best_acc = 0.
    best_epoch = None
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
        print('测试集1测试...')
        test_acc = test_accuracy(model, test_loader)
        print('Epoch [{}/{}] Loss: {:.4f} Train_Acc: {:.4f}  Test1_Acc: {:.4f}}'
              .format(epoch + 1, num_epochs, epoch_loss, train_acc, test_acc))
        scheduler.step(test_acc)
        if test_acc > best_acc:
            model_file = os.path.join(save_model_path, 'resnet34_CUB_200_fine_tuning_epoch_%d_acc_%g.pth' %
                                      (best_epoch, best_acc))
            if os.path.isfile(model_file):
                os.remove(os.path.join(save_model_path, 'resnet34_CUB_200_fine_tuning_epoch_%d_acc_%g.pth' %
                                       (best_epoch, best_acc)))
            end_patient = 0
            best_acc = test_acc
            best_epoch = epoch + 1
            print('测试集准确率提高，保存参数')
            torch.save(model.state_dict(), os.path.join(save_model_path,
                                                        'resnet34_CUB_200_fine_tuning_epoch_%d_acc_%g.pth' %
                                                        (best_epoch, best_acc)))
        else:
            end_patient += 1

        # 10次迭代测试集准确率不提高，则结束训练
        if end_patient >= 10:
            break
    print('训练结束,第%d个epoch结束,测试集准确率最高%g' % (best_epoch, best_acc))


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


def main():
    train()


if __name__ == '__main__':
    main()
