import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

#creates a session, initializes the variables,and evaluates, and f then closes the session
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)   #42
sess.close()

#instead of sess.run() everytime, do:
with tf.Session() as sess:
    x.initializer.run() # equivalent to calling tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval()   # equivalent to calling tf.get_default_session().run(f)

#manually calling every variable, use alt: global_variables_initializer()
init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()