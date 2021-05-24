import numpy as np
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from GoogLeNet import GoogLeNet
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
import torch



TRAIN_ROOT = './datasets/train'  # path for train datasets
TEST_ROOT = './datasets/test2'  # path for test datasets
PATH = './GoogLeNet.pth'  # path for load model

PRETRAINED_SIZE = 224  # define pretrained size
PRETRAINED_MEANS = [0.485, 0.456, 0.406]  # define pretrained means
PRETRAINED_STDS = [0.229, 0.224, 0.225]  # define pretrained stds
TRAIN_TRANSFORMS = transforms.Compose([  # define transforms for train datasets
        transforms.Resize(PRETRAINED_SIZE),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(PRETRAINED_SIZE, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=PRETRAINED_MEANS, std=PRETRAINED_STDS)])
TEST_TRANSFORMS = transforms.Compose([  # define transforms for test datasets
        transforms.Resize(PRETRAINED_SIZE),
        transforms.CenterCrop(PRETRAINED_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=PRETRAINED_MEANS, std=PRETRAINED_STDS)])
BATCH_SIZE = 40  # define batch size


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, predictions, normalize=True):
    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize=(15, 15))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]
        if normalize:
            image = normalize_image(image)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        predict = classes[predictions[i]]
        ax.set_title(f'Truth: {label} \n Predict: {predict}')
        ax.axis('off')
    plt.show()


def test_model(test_data, test_loader, model):
    """Function to test the model"""
    classes = test_data.classes
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            # plot_images(images, labels, classes, predictions)
            # break
            for label, prediction, image in zip(labels, predictions, images):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))


def train_model(data_loader, model, criterion, optimizer, path, epochs=1):
    """Function to train the model"""
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    torch.save(model.state_dict(), path)


def getTestData():
    test_data = datasets.ImageFolder(root=TEST_ROOT, transform=TEST_TRANSFORMS)
    return test_data


def getTestLoader():
    test_data = getTestData()
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return test_loader


def getTrainData():
    train_data = datasets.ImageFolder(root=TRAIN_ROOT, transform=TRAIN_TRANSFORMS)
    return train_data


def getTrainLoader():
    train_data = getTrainData()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader


def run(operation):
    model = GoogLeNet(aux_logits=False)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    if operation == 'test':
        # test model
        test_data = getTestData()
        test_loader = getTestLoader()
        print('start testing model...')
        test_model(test_data=test_data, test_loader=test_loader, model=model)
    elif operation == 'train':
        # train model
        # train_data = getTrainData()
        train_loader = getTrainLoader()
        train_model(data_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, path=PATH, epochs=1)


run(operation='test')


