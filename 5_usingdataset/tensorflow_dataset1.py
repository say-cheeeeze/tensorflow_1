import tensorflow as tf
import tensorflow_datasets as tfds

# /User/tensorflow_datasets 디렉토리에 다운로드된다.
fashion_mnist = 'fashion_mnist'
mnist_data, info = tfds.load(fashion_mnist, with_info=True)
# print('mnist_info ############')
# print(info)
# print('mnist_info ############')

'''
for item in mnist_data:
    print(item)  # train, test 2개가 있다.
'''

mnist_train_df = tfds.load(name=fashion_mnist, split='train')
assert isinstance(mnist_train_df, tf.data.Dataset)
# print(type(mnist_train_df))  # tensorflow.python.data.ops.dataset_ops.PrefetchDataset

'''
for item in mnist_train_df.take(1):
    print(type(item))  # dict
    print('item keys() : ', item.keys())  # dict_keys(['image', 'label'])
    print('item keys().image => ', type(item['image']))
    print(type(item['image']))
'''

(tr_img, tr_label), (test_img, test_label) = tfds.as_numpy(tfds.load(fashion_mnist,
                                                                     split=['train', 'test'],
                                                                     batch_size=-1,
                                                                     as_supervised=True)
                                                           )
# print( 'tr_img.shape : ' , tr_img.shape )
# print( 'tr_label.shape : ' , tr_label.shape )
# print( 'test_img.shape : ' , test_img.shape )
# print( 'test_label.shape : ' , test_label.shape )

training_img = tf.cast( tr_img, tf.float32 ) / 255.0



