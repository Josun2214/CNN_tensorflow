import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# 랜덤 시드 고정
tf.random.set_seed(10)

# 텐서보드 summary 정보들을 저장할 폴더 경로를 설정
train_summary_writer = tf.summary.create_file_writer('./tensorboard_log/train')
test_summary_writer = tf.summary.create_file_writer('./tensorboard_log/test')

# 하이퍼파라미터
batch_size = 128
learning_rate = 0.001
num_train_step = 5001

# CIFAR-10 데이터 다운로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# 데이터를 섞고 batch 단위로 묶기
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(batch_size)
train_data_iter = iter(train_data)

# 10번에 나눠서 추론(테스트) (메모리 초과 방지)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.repeat().batch(1000)
test_data_iter = iter(test_data)

# CNN 모델을 정의
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.conv_layer_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

        self.conv_layer_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

        self.conv_layer_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(384, activation='relu')

        # fc_layer_1 레이어에 드롭아웃 적용 (60%의 노드를 드롭아웃)
        self.dropout = tf.keras.layers.Dropout(0.6)

        # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(mapsping)합니다.
        self.output_layer = tf.keras.layers.Dense(10, activation=None)

    def call(self, x, is_training):
        h_conv1 = self.conv_layer_1(x)          # (128, 32, 32, 3)  ->  (128, 32, 32, 64)
        h_pool1 = self.pool_layer_1(h_conv1)    # (128, 32, 32, 64) ->  (128, 15, 15, 64)
        h_conv2 = self.conv_layer_2(h_pool1)    # (128, 15, 15, 64) ->  (128, 15, 15, 64)
        h_pool2 = self.pool_layer_2(h_conv2)    # (128, 15, 15, 64) ->  (128, 7, 7, 64)
        h_conv3 = self.conv_layer_3(h_pool2)    # (128, 7, 7, 64)   ->  (128, 7, 7, 128)
        h_conv4 = self.conv_layer_4(h_conv3)    # (128, 7, 7, 128)  ->  (128, 7, 7, 128)
        h_conv5 = self.conv_layer_5(h_conv4)    # (128, 7, 7, 128)  ->  (128, 7, 7, 128)
        h_flat = self.flatten_layer(h_conv5)    # (128, 7, 7, 128)  ->  (128, 6272)
        h_fc1 = self.fc_layer_1(h_flat)         # (128, 6272)       ->  (128, 384)
        # train 시 traingin=True, test 시 training=False (학습할 때만 드롭아웃 적용 후, 테스트시 드롭아웃 적용 x)
        h_fc1_drop = self.dropout(h_fc1, training=is_training)
        logits = self.output_layer(h_fc1_drop)  # (128, 384)        ->  (128, 10)
        y_pred = tf.nn.softmax(logits)

        return y_pred, logits

# cross-entropy 손실함수 정의
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 최적화를 위한 Adam 옵티마이저 정의
optimizer = tf.optimizers.Adam(learning_rate)

# 최적화를 위한 함수 정의 (실제 Gradient Descent를 수행하는 함수)
@tf.function
def train_step(model, x, y, is_training):
    with tf.GradientTape() as tape:
        _, logits = model(x, is_training=is_training)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 모델의 정확도를 출력하는 함수
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

# CNN 모델을 선언
CNN_model = CNN()

# num_train_step 만큼 반복을 수행하면서 최적화를 수행 (Gradient Descent 수행)
for i in range(num_train_step):
    batch_x, batch_y = next(train_data_iter)

    if i % 100 == 0:
        train_loss = cross_entropy_loss(CNN_model(batch_x, is_training=False)[1], batch_y)
        train_accuracy = compute_accuracy(CNN_model(batch_x, is_training=False)[0], batch_y)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=optimizer.iterations)
            tf.summary.scalar('accuracy', train_accuracy, step=optimizer.iterations)

        ####
        test_accuracy = 0.0
        for _ in range(10):
            test_batch_x, test_batch_y = next(test_data_iter)
            test_loss = cross_entropy_loss(CNN_model(test_batch_x, is_training=False)[1], test_batch_y)
            test_accuracy += compute_accuracy(CNN_model(test_batch_x, is_training=False)[0], test_batch_y).numpy()
        test_accuracy /= 10
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss, step=optimizer.iterations)
            tf.summary.scalar('accuracy', test_accuracy, step=optimizer.iterations)
        ####

        print(f'Epoch {i} - train_loss: {train_loss}\t\ttrain_accuracy: {train_accuracy}')

    train_step(CNN_model, batch_x, batch_y, is_training=True)
