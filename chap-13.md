# Convolutional Neural Networks

human eyes respond to visual
stimuli located in a limited region of the visual field

some neurons respond only to horizontal, some only to vertical

## [Convolution Layer](./cnn-convolution.py)
neurons in the first convolutional layer are not connected to every single pixel in the input image, but only to pixels in their receptive fields 

A neuron located in row i, column j of a given layer is connected to the outputs of the
neurons in the previous layer located in rows i to i + fh – 1, columns j to j + fw – 1,
where fh and fw are the height and width of the receptive field

*zero-padding*: for a layer to have the same height and width as the previous layer, it is common to add zeros around the inputs

*stride*: distance between two consecutive receptive fields

*Filter*: Applying vertical and horizontal filters creates separate feature maps, and those maps help the network understand structured patterns (like edges) at a deeper level.

### Stacking Multiple Feature Maps
a convolutional layer simultaneously applies multiple filters to its inputs, making it capable of detecting multiple features anywhere in its inputs.

Within one feature map, all neurons share the same parameters (weights and bias term), but different feature maps may have different parameters

neuron in row i, column j of feature map k in layer l is connected to l-1, in rows i × sw to i × sw + fw – 1 and columns j × sh to j × sh + fh – 1

output of neuron in a convolutional layer

$$
z_{i,j,k} = b_k + \sum_{u=1}^{f_h} \sum_{v=1}^{f_w} \sum_{k'=1}^{f_{n'}} x_{i',j',k'} \cdot w_{u,v,k',k} \quad \text{with} \begin{cases} i' = u \cdot s_h + f_h - 1 \\ j' = v \cdot s_w + f_w - 1 \end{cases}
$$

where:
- $z_{i,j,k}$ — output of neuron at row $i$, column $j$, feature map $k$ (layer $l$)
- $s_h, s_w$ — vertical and horizontal strides
- $f_h, f_w$ — height and width of the receptive field
- $f_{n'}$ — number of feature maps in the previous layer ($l-1$)
- $x_{i',j',k'}$ — output of neuron in layer $l-1$ at row $i'$, column $j'$, feature map $k'$
- $b_k$ — bias term for feature map $k$ (tweaks overall brightness of that map)
- $w_{u,v,k',k}$ — connection weight between any neuron in feature map $k$ of layer $l$ and its input at row $u$, column $v$ (relative to receptive field), feature map $k'$

implementation:
```python
import numpy as np
from sklearn.datasets import load_sample_images

# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 0] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))   #input mini-batch
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")   #set of filters to apply 
#padding=SAME-> 0 padding; padding=VALID-> may ignore some rows and columns

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.show()
```

## [Pooling Layer](./cnn-pooling.py)
subsample (i.e., shrink) the input image in order to reduce the computational load, the memory usage, and the number of parameters

each neuron in a pooling layer is connected to the
outputs of a limited number of neurons in the previous layer, located within a small
rectangular receptive field.

2 × 2 pooling kernel, a stride of 2, and no padding.
```python
# Create a graph with input X plus a max pooling layer
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})
plt.imshow(output[0].astype(np.uint8)) # plot the output for the 1st image
plt.show()

```

drops upto 75% of input value

## CNN Architecture
stack a few convolutional layers (each one generally fol‐
lowed by a ReLU layer), then a pooling layer, then another few convolutional layers
(+ReLU), then another pooling layer, and so on

images gets smaller at each step

3 main architectures:


**[LeNet-5 Architecture(1998):](./le-net5.py)**

| Layer | Type | Maps | Size | Kernel | Stride | Activation |
|-------|------|------|------|--------|--------|------------|
| In | Input | 1 | 32×32 | – | – | – |
| C1 | Convolution | 6 | 28×28 | 5×5 | 1 | tanh |
| S2 | Avg Pooling | 6 | 14×14 | 2×2 | 2 | tanh |
| C3 | Convolution | 16 | 10×10 | 5×5 | 1 | tanh |
| S4 | Avg Pooling | 16 | 5×5 | 2×2 | 2 | tanh |
| C5 | Convolution | 120 | 1×1 | 5×5 | 1 | tanh |
| F6 | Fully Connected | – | 84 | – | – | tanh |
| Out | Fully Connected | – | 10 | – | – | RBF |

**[AlexNet Architecture:](./alex-net.py)**

| Layer | Type | Maps | Size | Kernel | Stride | Padding | Activation |
|-------|------|------|------|--------|--------|---------|------------|
| In | Input | 3 (RGB) | 224×224 | – | – | – | – |
| C1 | Convolution | 96 | 55×55 | 11×11 | 4 | SAME | ReLU |
| S2 | Max Pooling | 96 | 27×27 | 3×3 | 2 | VALID | – |
| C3 | Convolution | 256 | 27×27 | 5×5 | 1 | SAME | ReLU |
| S4 | Max Pooling | 256 | 13×13 | 3×3 | 2 | VALID | – |
| C5 | Convolution | 384 | 13×13 | 3×3 | 1 | SAME | ReLU |
| C6 | Convolution | 384 | 13×13 | 3×3 | 1 | SAME | ReLU |
| C7 | Convolution | 256 | 13×13 | 3×3 | 1 | SAME | ReLU |
| F8 | Fully Connected | – | 4,096 | – | – | – | ReLU |
| F9 | Fully Connected | – | 4,096 | – | – | – | ReLU |
| Out | Fully Connected | – | 1,000 | – | – | – | Softmax |

**Local Response Normalization (LRN)** — used in AlexNet after ReLU

$$
b_i = a_i \left( k + \alpha \sum_{j=j_{\text{low}}}^{j_{\text{high}}} a_j^2 \right)^{-\beta} \quad \text{with} \begin{cases} j_{\text{high}} = \min(i + r/2,\; f_n - 1) \\ j_{\text{low}} = \max(0,\; i - r/2) \end{cases}
$$

where:
- $b_i$ — normalized output of neuron in feature map $i$ (at some row $u$, col $v$)
- $a_i$ — activation after ReLU, before normalization
- $k$ (bias), $\alpha$, $\beta$, $r$ (depth radius) — hyperparameters
- $f_n$ — number of feature maps
- AlexNet uses: $r=2$, $\alpha=0.00002$, $\beta=0.75$, $k=1$

if $r=2$ and a neuron has strong activation → it inhibits neurons in feature maps immediately above and below (competitive normalization)

ZF Net (2013 winner) — essentially AlexNet with tweaked hyperparameters (feature maps, kernel size, stride)

---

### [GoogLeNet (2014)](./google-net.py)

won ILSVRC 2014, top-5 error rate below 7%
much deeper than previous CNNs but 10× fewer parameters than AlexNet (~6M vs ~60M)

**Inception Module** — the key building block:
- input is copied and fed to 4 parallel paths:
  1. 1×1 conv
  2. 1×1 conv → 3×3 conv
  3. 1×1 conv → 5×5 conv
  4. 3×3 max pool → 1×1 conv
- all use stride 1, SAME padding → same height/width
- outputs concatenated along depth dimension (`tf.concat(axis=3)`)
- all conv layers use ReLU

**Why 1×1 convolutions?**
1. **bottleneck layer** — reduces dimensionality (fewer feature maps than input), especially useful before expensive 3×3 and 5×5 convs
2. **two-layer power** — pair of [1×1, 3×3] acts like sweeping a 2-layer neural network across the image, capturing more complex patterns

**GoogLeNet architecture flow:**
1. 2 conv layers → reduce height/width by 4× (area ÷16)
2. LRN → learn wide variety of features
3. 2 conv layers (1×1 bottleneck + 3×3) — acts as one smart conv layer
4. LRN again
5. max pool → reduce by 2×
6. 9 inception modules (interleaved with max pooling)
7. **global average pooling** — kernel = feature map size, VALID padding → 1×1 output per map
   - forces previous layers to produce confidence maps for each class
   - eliminates need for multiple FC layers → fewer parameters, less overfitting
8. dropout → FC layer with softmax

original also had 2 auxiliary classifiers (at 3rd and 6th inception module) — loss scaled by 0.7, added to main loss to fight vanishing gradients (minor effect)

---

### [ResNet (2015)](./res-net.py)

won ILSVRC 2015, top-5 error rate under 3.6%, 152 layers deep

**Key idea: skip connections (shortcut connections)**
- signal feeding into a layer is added to output of a layer higher up
- network learns $f(x) = h(x) - x$ (residual) instead of $h(x)$ directly → **residual learning**

**Why it works:**
1. weights init near zero → without skip: output ≈ 0; with skip: output ≈ input (identity function)
2. if target ≈ identity (often the case) → training is much faster
3. signal can bypass layers that haven't learned yet → gradient flows easily across the whole network

**Residual unit** = small neural network + skip connection
- 2 conv layers (3×3, stride 1, SAME, BN + ReLU)
- preserves spatial dimensions

**ResNet architecture:**
- starts and ends like GoogLeNet (no dropout)
- deep stack of residual units in between
- feature maps double every few RUs, height/width halve (conv with stride 2)
- when dimensions change → skip connection uses 1×1 conv with stride 2 to match shape

**ResNet-34:**
- 3 RUs × 64 maps → 4 RUs × 128 maps → 6 RUs × 256 maps → 3 RUs × 512 maps

**ResNet-152** (deeper variant):
- uses 3 conv layers per RU instead of 2: 1×1 (bottleneck, ÷4) → 3×3 → 1×1 (restore depth)
- 3 RUs × 256 → 8 RUs × 512 → 36 RUs × 1024 → 3 RUs × 2048

**Ranking:** ResNet > GoogLeNet > VGGNet > AlexNet > LeNet-5

other notable architectures:
- **VGGNet** (2014 runner-up) — simple but deep (16-19 layers, all 3×3 convs)
- **Inception-v4** — merges GoogLeNet + ResNet ideas, ~3% top-5 error

----

[using them all](./cnn-final.py)