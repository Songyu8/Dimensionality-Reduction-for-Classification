import gzip
import  struct   
import os
import numpy as np
import pickle
def unpickle(file):
    '''  Un-pack data '''
    import pickle
    with open(file, 'rb') as fo:
        ds = pickle.load(fo, encoding='bytes')
    return ds

def print_dataset_statistics(train_labels,test_labels):

    print("Dataset statistics:")
    print("  ----------------------------------------")
    print("  subset   | # images | # classes | ")
    print("  ----------------------------------------")
    print("  train    | {:5d}    | {:8d}  | ".format(len(train_labels), 10))
    print("  test     | {:5d}    | {:8d}  | ".format(len(test_labels), 10))
    print("  ----------------------------------------")

    print('DATA IS LOADING DOWN')


def test_CIFAR10():
    path = 'C:\\Users\\songyu\\Desktop\\Cifar_10-main\\cifar-10-batches-py\\'
    x=[]
    y=[]
    files_path = os.listdir(path)

    train_patches = files_path[1:6]
    test_patch = files_path[7]
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    num_train, num_test = 50000, 10000
    class_names = ['airplane', 'automobile', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for data_batch in train_patches:
        batch_dict=unpickle(path+data_batch)
        train_batch=batch_dict[b'data'].astype('float')
        train_labels=np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_labels)
    #将5个训练样本batch合并为50000x3072，标签合并为50000x1
    #np.concatenate默认axis=0，为纵向连接
    train_data=np.concatenate(x)
    train_labels=np.concatenate(y)

    test_dict=unpickle(path+test_patch)
    test_data=test_dict[b'data'].astype('float')
    test_labels=np.array(test_dict[b'labels'])

    print_dataset_statistics(train_labels,test_labels)

    return train_data,train_labels,test_data,test_labels

# ------------------------------------------------------------------------------------------------------------------------
def test_EXMNIST():
    path='C:\\Users\\songyu\\Desktop\\gzip'

    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% 'emnist-mnist-train')
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% 'emnist-mnist-train')
    #使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        train_labels = np.frombuffer(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        train_data = np.frombuffer(imgpath.read(),dtype=np.uint8).reshape(len(train_labels), 784)


    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% 'emnist-mnist-test')
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% 'emnist-mnist-test')
    #使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        test_labels = np.frombuffer(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        test_data = np.frombuffer(imgpath.read(),dtype=np.uint8).reshape(len(test_labels), 784)

    print_dataset_statistics(train_labels,test_labels)
    
    return train_data,train_labels,test_data,test_labels

