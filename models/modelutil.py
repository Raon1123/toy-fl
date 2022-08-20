from torchvision.models import resnet18

from models.cnn import CNN, NaiveCNN

def get_model(args, num_classes, in_channel):
    if args.model == 'CNN':
        model = CNN(num_classes=num_classes, in_channel=in_channel)
    elif args.model == 'ResNet18':
        model = resnet18(num_classes=num_classes)
    elif args.model == 'NaiveCNN':
        model = NaiveCNN(args=args, input_shape = [3, 32, 32], num_classes=num_classes, final_pool=False)
    else:
        Exception('Wrong model definition') 

    print(model)
    return model