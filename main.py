from utils import *
from models import *
import argparse
# main.py

import argparse

def main():
    parser = argparse.ArgumentParser(description='CRC Image Classification')

    # train subcommand
    parser_cmd = parser.add_subparsers(dest='command', help='Available commands')
    train_cmd = parser_cmd.add_parser('train', help='Train the model')  # 创建一个命令对象
    train_cmd.add_argument('--model', type=str, default='ResNet34', help='Type of the model used')
    train_cmd.add_argument('--data', type=str, default='data/CRC-VAL-HE-7K/', help='Path to the processed data directory')
    train_cmd.add_argument('--dataset', type=str, default='CRC-VAL-HE-7K', help='Path to the dataset directory')
    train_cmd.add_argument('--log', type=str, default='log/', help='Path to save trained models and training logs')
    train_cmd.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    train_cmd.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    train_cmd.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for training')
    train_cmd.add_argument('--pretrained', type=bool, default=False, help='Use pretrained parameters')
    train_cmd.add_argument('--num_classes', type=int, default=9, help='Number of the classes')

    # predict subcommand
    predict_cmd = parser_cmd.add_parser('predict', help='Predict using the trained model')
    predict_cmd.add_argument('--img', type=str, default="data\example.tif", help='Path to the image for prediction')
    predict_cmd.add_argument('--num_classes', type=int, default=9, help='Number of the classes')
    predict_cmd.add_argument('--model', type=str, default='ResNet34', help='Type of the model used')
    predict_cmd.add_argument('--dataset', type=str, default='CRC-VAL-HE-7K', help='Path to the dataset directory')
    
    args = parser.parse_args()
    
    # Initialize and train the ResNet model
    if args.model == 'ResNet34' or args.model == 'ResNet':
        net = ResNet(BasicBlock, (16,32,64,128), (3,4,6,3), num_classes=args.num_classes, type=34)
        pt = "models\ResNet34_CRC.pt"
    elif args.model == 'ResNet50':
        net = ResNet(BottleNeck, (16,32,64,128), (3,4,6,3), num_classes=args.num_classes, type=50)
        pt = ""
    elif args.model == 'ViT':
        net = ViT(num_classes=args.num_classes)
        pt = "models\ViT_CRC.pt"
    else:
        print("ERROR: No Such Model")
        return
    net = net.to(device)
    if args.command == 'predict':
        label = predict(model=args.model, net=net, pt=pt, dataset=args.dataset, img=args.img)
    elif args.command == 'train':
        train(model=args.model, net=net, data=args.data, learning_rate=args.learning_rate, batch=args.batch_size, epoch=args.epochs, num_classes=args.num_classes, pretrained=args.pretrained, log=args.log)
    else:
        train(model=args.model, net=net, data=args.data, learning_rate=args.learning_rate, batch=args.batch_size, epoch=args.epochs, num_classes=args.num_classes, pretrained=args.pretrained, log=args.log)

if __name__ == '__main__':
    main()

