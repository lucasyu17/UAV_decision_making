# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
from UAV import UAV
# import pcl

TIME_STEP = 32                                              #每一个图片的分层：32层
img_width = 64
INPUT_SIZE = img_width*img_width                            #每一层的一张图片大小
LR = 0.01                                                   #学习率
NUM_UNITS = 128                                             #多少个lstm学习单元
ITERATIONS = 100                                             #迭代次数
N_CLASSES = 6                                               #输出的维度
batch_step = 20                                             #一个batch包含多少张图片
total_imgNum = 1000                                         #数据中总共有多少张图片

weights = {
    'in':tf.Variable(tf.random_normal([INPUT_SIZE,NUM_UNITS]),name='w_in'),
    'out':tf.Variable(tf.random_normal([NUM_UNITS,N_CLASSES]),name='w_out')
}
biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[NUM_UNITS,]),name='b_in'),
    'out':tf.Variable(tf.constant(0.1,shape=[N_CLASSES,]),name='b_out')
}

uav = UAV()
uav.read_pcd()
uav.get_global_pos()
uav.printCurPos()

saver = tf.train.Saver(max_to_keep=4)

class RNN_DATA:
    def __init__(self,X,Y):  #X: （total_imgNum*TIME_STEP,64*64）  Y: （total_imgNum,4）
        self.x = X
        self.y = Y
        self.checkPoint = 0
        self.ckpts_y = 0
    def get_next_batch(self,batch_step):
        self.checkPoint += batch_step
        self.ckpts_y += batch_step
        print(self.checkPoint)
        if self.checkPoint < total_imgNum:
            x_tmp,y_tmp = self.x[TIME_STEP*(self.checkPoint-batch_step):self.checkPoint*TIME_STEP,:], \
                          self.y[self.ckpts_y-batch_step:self.ckpts_y]
            return x_tmp.reshape([batch_step,TIME_STEP,INPUT_SIZE]),y_tmp.reshape([batch_step,N_CLASSES])
        else:
            return False,False

    def chgDataFormat(self):
        #read img data and change format into arrays
        return 0

#train data container
train_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE]) # 维度是[BATCH_SIZE，TIME_STEP * INPUT_SIZE]
train_y = tf.placeholder(tf.int32, [None, N_CLASSES])

#test data container
test_x = tf.placeholder(tf.float32,shape=[],name = 'test_x')
test_y = tf.placeholder(tf.float32,shape=[],name = 'test_y')

def fake_data():
    fake_img = np.random.randint(1,size=(total_imgNum*TIME_STEP,img_width*img_width))
    rand_posHuman = [np.random.randint(low=5,high=60,size=2) for i in range(total_imgNum)]
    fake_y = np.zeros(shape=[total_imgNum,4])
    for i in range(3):
        fake_img[rand_posHuman[0]+i,rand_posHuman[1]+i] = 2
    fake_x = fake_img

    for i in range(len(rand_posHuman)):
        if rand_posHuman[i][1]>32:
            fake_y[i,0] = 1

    return fake_x,fake_y

def load_model(load_pt):
    with tf.Session() as sess_test:
        saver = tf.train.import_meta_graph('model/my-model-'+str(load_pt)+'.meta')
        saver.restore(sess_test, tf.train.latest_checkpoint("model/"))
        print sess_test.run('w_in:0')
        print sess_test.run('b_in:0')
        return sess_test

def save_model(sess,step):
    saver.save(sess, "model/my-model", global_step=step)

# 定义RNN（LSTM）结构
def RNN(X,weights,biases):

    # hidden unit for input #
    X = tf.reshape(X,[-1,INPUT_SIZE])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in,[batch_step,TIME_STEP,NUM_UNITS])

    # cell #
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)
    _init_state = rnn_cell.zero_state(batch_step,dtype=tf.float32)

    outputs,states = tf.nn.dynamic_rnn(
    cell=rnn_cell,              # 选择传入的cell
    inputs=X_in,                # 传入的数据
    initial_state=_init_state,  # 初始状态
    dtype=tf.float32,           # 数据类型
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False

    )

    # hidden layer for output #
    results = tf.matmul(states[1],weights['out'])+biases['out']
    return results

fakex,fakey = fake_data()
print(np.sum(fakey))

rnn_data = RNN_DATA(fakex,fakey)


pred = RNN(train_x,weights,biases)
loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=pred)              # 计算loss
train_op = tf.train.AdamOptimizer(LR).minimize(loss)                                    #选择优化方法
correct_prediction = tf.equal(tf.argmax(pred, axis=1),tf.argmax(train_y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))                          #计算正确率
sess = tf.Session()

sess.run(tf.global_variables_initializer())                                             #初始化计算图中的变量

def train():
    for step in range(ITERATIONS):    # 开始训练
        print("training\n")
        x,y = rnn_data.get_next_batch(batch_step)
        if(type(x)==bool and x==False):
            print("All data trained")
            break
        else:
            x=x.reshape(batch_step,TIME_STEP,INPUT_SIZE)
            test_x = np.random.rand(640, 64 * 64)
            test_y = np.zeros(shape=(batch_step, N_CLASSES))
            rand_pos = np.random.randint(2)
            if rand_pos>1:
                test_y[0] = 1
            test_x = test_x.reshape(batch_step, TIME_STEP, INPUT_SIZE)
            _, loss_ = sess.run([train_op, loss], {train_x: x, train_y: y})

            if step % 2 == 0:      # test（validation）
                accuracy_ = sess.run(accuracy, {train_x: test_x, train_y: test_y})
                print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
            if step % 10 ==0:
                save_model(sess, step)
    return True
def test(checkpoint):
    load_model(checkpoint)
    return True

train()
sess = test(20)