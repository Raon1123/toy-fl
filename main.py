import os

from utils.parser import cifar10_dict, argparser

def main():
    args = argparser()

    data_PATH = os.path.join(args.data_dir, 'cifar-10-batches-py')
    dic = cifar10_dict(data_PATH)
    print(dic)


if __name__=='__main__':
    main()