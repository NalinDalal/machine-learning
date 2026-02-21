# Distributing TensorFlow Across Devices and Servers
break down the whole training via parallelisation to save your time

# Multiple Devices on a Single Machine

Install cuda, cuda is like helps tensorflow talk with gpu
install gpu version of tensorflow
`pip3 install --upgrade tensorflow-gpu`

## Managing GPU RAM
when 1st time u run the graph, 2nd can't be started
tell TensorFlow to grab only a fraction of the memory.(say 40%)
```python
#log placement constraints
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
x.initializer.run(session=sess)
sess.run(c) #12

#dynamic placement function: helps to call custom gpu
def variables_on_cpu(op):
    if op.type == "Variable":
        return "/cpu:0"
    else:
        return "/gpu:0"

with tf.device(variables_on_cpu):
    a = tf.Variable(3.0)
    b = tf.constant(4.0)
    c = a * b
```

## Parallel Execution
u know how parallel execution works
TensorFlow manages a thread pool on each device to parallelize operations. 
These are called the inter-op thread pools. Some operations have multi‐threaded kernels: 
they can use other thread pools (one per device) called the intra-op thread pools.

## Control Dependencies
evaluate x,y only after a,b is evaluated:
```python
a = tf.constant(1.0)
b = a + 2.0
with tf.control_dependencies([a, b]):
    x = tf.constant(3.0)
    y = tf.constant(4.0)
z = x + y
```

# Multiple Devices Across Multiple Servers

define a cluster to run a graph across multiple server
cluster: multiple tenserflow server known as Tasks
Each task belongs to a job. 

```python
cluster_spec = tf.train.ClusterSpec({
"ps": [ #job 1
"machine-a.example.com:2221", # /job:ps/task:0
],
"worker": [ #job2
"machine-a.example.com:2222", # /job:worker/task:0
"machine-b.example.com:2222", # /job:worker/task:1
]})

#create a server; create a Server object, passing it the cluster specification and its own job name and task number. 
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)

server.join() # blocks until the server stops (i.e., never)
#block the main thread by telling it to wait for the server to finish
```

## Opening a Session
tasks are up, so can start a session on any server
```python
a = tf.constant(1.0)
b = a + 2
c = a * 3
with tf.Session("grpc://machine-b.example.com:2222") as sess:
    print(c.eval()) # 9.0
```

## Master and Worker Services
client uses grpc to talk with server
Data is transmitted in the form of protocol buffers(lightweight binary data interchange format.)
2 services: master & worker

master: client talks with it, tells what to do
worker: actually does the tasks

## Pinning Operations Across Tasks
use device blocks to pin operations on any device managed by any task, by
specifying the job name, task index, device type, and device index
```python
with tf.device("/job:ps/task:0/cpu:0")
    a = tf.constant(1.0)
with tf.device("/job:worker/task:0/gpu:1")
    b = a + 2
c = a + b
```

## Sharding Variables Across Multiple Parameter Servers
useful to shard these parameters across multiple parameter servers, to reduce the risk of saturating a single parameter server’s network card

replica_device_setter() function, which distributes variables across all the "ps" tasks in a round-robin fashion.

```python
with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
    v1 = tf.Variable(1.0) # pinned to /job:ps/task:0
    v2 = tf.Variable(2.0) # pinned to /job:ps/task:1
    v3 = tf.Variable(3.0) # pinned to /job:ps/task:0
    v4 = tf.Variable(4.0) # pinned to /job:ps/task:1
    v5 = tf.Variable(5.0) # pinned to /job:ps/task:0

with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
    v1 = tf.Variable(1.0) # pinned to /job:ps/task:0 (+ defaults to /cpu:0)
    v2 = tf.Variable(2.0) # pinned to /job:ps/task:1 (+ defaults to /cpu:0)
    v3 = tf.Variable(3.0) # pinned to /job:ps/task:0 (+ defaults to /cpu:0)
    [...]
    s = v1 + v2 # pinned to /job:worker (+ defaults to task:0/gpu:0)
    with tf.device("/gpu:1"):
        p1 = 2 * s # pinned to /job:worker/gpu:1 (+ defaults to /task:0)
        with tf.device("/task:1"):
            p2 = 3 * s # pinned to /job:worker/task:1/gpu:1
```

## Sharing State Across Sessions Using Resource Containers
each session has own memory, to let them share memory:
```python
# simple_client.py
import tensorflow as tf
import sys

x = tf.Variable(0.0, name="x")
increment_x = tf.assign(x, x + 1)

with tf.Session(sys.argv[1]) as sess:
    if sys.argv[2:]==["init"]:
        sess.run(x.initializer)
    sess.run(increment_x)
    print(x.eval())
```

to clear up the resources: `tf.Session.reset("grpc://machine-a.example.com:2222", ["my_problem_1"])`

## Asynchronous Communication Using TensorFlow Queues
uses queues to load and consume data
```python
#queue creation
q = tf.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[[2]],name="q", shared_name="shared_q")

#enqueuing data; enqueue the three tensors to the queue
import tensorflow as tf
q = [...]
training_instance = tf.placeholder(tf.float32, shape=(None,2))
enqueue = q.enqueue([training_instance])
with tf.Session("grpc://machine-a.example.com:2222") as sess:
    sess.run(enqueue_many, feed_dict={training_instances: [[1., 2.],[3., 4.], [5., 6.]]})

#dequeuing data
q = [...]
dequeue = q.dequeue()
with tf.Session("grpc://machine-a.example.com:2222") as sess:
    print(sess.run(dequeue)) # [1., 2.]
    print(sess.run(dequeue)) # [3., 4.]
    print(sess.run(dequeue)) # [5., 6.]

#or dequque at once
[...]
batch_size = 2
dequeue_mini_batch= q.dequeue_many(batch_size)
with tf.Session("grpc://machine-a.example.com:2222") as sess:
    print(sess.run(dequeue_mini_batch)) # [[1., 2.], [4., 5.]]
    print(sess.run(dequeue_mini_batch)) # blocked waiting for another instance

#queue of tuple
q = tf.FIFOQueue(capacity=10, dtypes=[tf.int32, tf.float32], shapes=[[],[3,2]],name="q", shared_name="shared_q")

a = tf.placeholder(tf.int32, shape=())
b = tf.placeholder(tf.float32, shape=(3, 2))
enqueue = q.enqueue((a, b))

with tf.Session([...]) as sess:
    sess.run(enqueue, feed_dict={a: 10, b:[[1., 2.], [3., 4.], [5., 6.]]})
    sess.run(enqueue, feed_dict={a: 11, b:[[2., 4.], [6., 8.], [0., 2.]]})
    sess.run(enqueue, feed_dict={a: 12, b:[[3., 6.], [9., 2.], [5., 8.]]})

#dequque them
dequeue_a, dequeue_b = q.dequeue()
with tf.Session([...]) as sess:
    a_val, b_val = sess.run([dequeue_a, dequeue_b])
    print(a_val) # 10
    print(b_val) # [[1., 2.], [3., 4.], [5., 6.]]

#queue of queue
q = tf.FIFOQueue(capacity=10, dtypes=[tf.int32, tf.float32], shapes=[[],[3,2]],name="q", shared_name="shared_q")
a = tf.placeholder(tf.int32, shape=())
b = tf.placeholder(tf.float32, shape=(3, 2))
enqueue = q.enqueue((a, b))
with tf.Session([...]) as sess:
    sess.run(enqueue, feed_dict={a: 10, b:[[1., 2.], [3., 4.], [5., 6.]]})
    sess.run(enqueue, feed_dict={a: 11, b:[[2., 4.], [6., 8.], [0., 2.]]})
    sess.run(enqueue, feed_dict={a: 12, b:[[3., 6.], [9., 2.], [5., 8.]]})

dequeue_a, dequeue_b = q.dequeue()

#or 
with tf.Session([...]) as sess:
    a_val, b_val = sess.run([dequeue_a, dequeue_b])
    print(a_val) # 10
    print(b_val) # [[1., 2.], [3., 4.], [5., 6.]]

#closing a queue
close_q = q.close()
with tf.Session([...]) as sess:
    [...]
    sess.run(close_q)

#RandomShuffleQueue: dequeue in random order
dequeue = q.dequeue_many(5)
with tf.Session([...]) as sess:
    print(sess.run(dequeue)) # [ 20. 15. 11. 12. 4.] (17 items left)
    print(sess.run(dequeue)) # [ 5. 13. 6. 0. 17.] (12 items left)
    print(sess.run(dequeue)) # 12 - 5 < 10: blocked waiting for 3 more instances

# PaddingFIFOQueue: accepts tensors of variable sizes along any dimension
q = tf.PaddingFIFOQueue(capacity=50, dtypes=[tf.float32], shapes=[(None, None)] name="q", shared_name="shared_q")
v = tf.placeholder(tf.float32, shape=(None, None))
enqueue = q.enqueue([v])
with tf.Session([...]) as sess:
    sess.run(enqueue, feed_dict={v: [[1., 2.], [3., 4.], [5., 6.]]}) # 3x2
    sess.run(enqueue, feed_dict={v: [[1.]]}) # 1x1
    sess.run(enqueue, feed_dict={v: [[7., 8., 9., 5.], [6., 7., 8., 9.]]}) # 2x4
```

## Loading Data from Graph

**preload data into a variable**
```python
training_set_init = tf.placeholder(tf.float32, shape=(None, n_features))
training_set = tf.Variable(training_set_init, trainable=False, collections=[],name="training_set")
with tf.Session([...]) as sess:
    data = [...] # load the training data from the datastore
    sess.run(training_set.initializer, feed_dict={training_set_init: data})
```

**Reading the training data directly from the graph**
```python
reader = tf.TextLineReader(skip_header_lines=1)
filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

key, value = reader.read(filename_queue)

x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
features = tf.stack([x1, x2])

instance_queue = tf.RandomShuffleQueue(
capacity=10, min_after_dequeue=2,
dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
name="instance_q", shared_name="shared_instance_q")
enqueue_instance = instance_queue.enqueue([features, target])
close_instance_queue = instance_queue.close()

#running it
with tf.Session([...]) as sess:
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    try:
        while True:
            sess.run(enqueue_instance)
    except tf.errors.OutOfRangeError as ex:
        pass # no more records in the current file and no more files to read
    sess.run(close_instance_queue)

# create the shared instance queue and simply dequeue mini-batches from it
instance_queue = tf.RandomShuffleQueue([...], shared_name="shared_instance_q")
mini_batch_instances, mini_batch_targets = instance_queue.dequeue_up_to(2)
[...] # use the mini_batch instances and targets to build the training graph
training_op = [...]

with tf.Session([...]) as sess:
    try:
        for step in range(max_steps):
            sess.run(training_op)
    except tf.errors.OutOfRangeError as ex:
        pass # no more training instances
```

**Multithreaded readers using a Coordinator and a QueueRunner**
```python
#create threads and manage yourself
coord = tf.train.Coordinator()

while not coord.should_stop():
    [...] # do something

coord.request_stop()

coord.join(list_of_threads)

queue_runner = tf.train.QueueRunner(instance_queue, [enqueue_instance] * 5)
with tf.Session() as sess:
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    coord = tf.train.Coordinator()
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)

# create a reader and the nodes that will read and push one instance to the instance queue:
def read_and_push_instance(filename_queue, instance_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
    features = tf.stack([x1, x2])
    enqueue_instance = instance_queue.enqueue([features, target])
    return enqueue_instance

#define the queues
filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

instance_queue = tf.RandomShuffleQueue([...])

read_and_enqueue_ops = [
    read_and_push_instance(filename_queue, instance_queue)
    for i in range(5)]
queue_runner = tf.train.QueueRunner(instance_queue, read_and_enqueue_ops)
```

# Parallelizing Neural Networks on a TensorFlow Cluster

**One Neural Network per Device**
## in-graph replication: 
You build **one giant computational graph** that contains:

* Model 1 (on GPU 0)
* Model 2 (on GPU 1)
* Model 3 (on GPU 2)
* Aggregation logic

All inside a single TensorFlow graph.

Everything lives in one graph.
One session runs it.
TensorFlow coordinates internally.

1. Input comes in
2. Each model computes prediction in parallel
3. Graph waits for all predictions
4. Aggregation happens
5. Final output returned

| Advantages| Disadvantages |
|------|------|
| Simple|Large graph becomes complex|
| Single session| Harder to test each model independently|
| TensorFlow handles coordination|Less fault tolerance (if one model crashes → whole graph fails)|
| Easier initial implementation ||

## Between-Graph Replication:
You create **one graph per model**.
Each model:

* Runs independently
* Has its own session
* Has its own input and output queue

 manual coordination


Here we have:

* Client A → Model 1
* Client B → Model 2
* Client C → Model 3
* Aggregator Client → reads all predictions and combines

**Flow:**
1. Input distributor copies input to all input queues
2. Each model:

   * Reads from its queue
   * Produces prediction
   * Writes to its prediction queue
3. Aggregator:

   * Reads one prediction from each queue
   * Aggregates

**Advantages:**
- Modular
- Easier to test each model
- Better fault tolerance
- More flexible
- Can add timeouts



--

In-Graph =
One giant factory with many machines inside one building.

Between-Graph =
Separate factories communicating via conveyor belts.

If one machine breaks:

* In-Graph → whole factory halts
* Between-Graph → only one factory fails, others keep working

---

Between-graph replication is conceptually closer to:

* Microservices architecture
* Distributed systems
* Kubernetes pods
* Independent model serving

In-graph is closer to:

* Single-process multi-threaded program




| Feature              | In-Graph         | Between-Graph |
| -------------------- | ---------------- | ------------- |
| Implementation       | Easier           | More setup    |
| Fault tolerance      | Weak             | Strong        |
| Modularity           | Low              | High          |
| Scalability          | Limited by graph | More flexible |
| Testing              | Harder           | Easier        |
| Production readiness | OK               | Better        |


| InGraph | BetweenGraph |
|---------|--------------|
|Research setup|Production system|
|Small ensemble|Large cluster|
|Controlled environment|Need robustness|
|Simpler infrastructure|Want isolation between models|