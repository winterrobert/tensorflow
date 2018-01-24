# https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

#define variables
node3 = tf.constant(3.0)
node5 = tf.constant(5.0);
a = tf.placeholder(tf.float32)
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

#define methods
node_add = node3 + node5
adder_node = a + b
linear_model = W*x + b

sess = tf.Session()
# init variables
init = tf.global_variables_initializer()
sess.run(init)

#print stuff
print(sess.run(node_add))
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

#create error function and calculate loss
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))

#set correct thetas by training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) #reset global variables
for i in range(1000):
	sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})

#print the modified variables/thetas
print(sess.run([W,b]))