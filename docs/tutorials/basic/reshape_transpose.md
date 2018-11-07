
## Difference between reshape and transpose operators
Modyfing the shape of tensors is a very common operation in Deep Learning. For instance, when using pretrained neural networks it is often required to adjust input data dimensions to correspond to what the network has been trained on, e.g. tensors of shape `[batch_size, channels, width, height]`.  This notebook discusses briefly the difference between the operators `Reshape` and `Transpose`. Both allow to change the shape, however they are not the same and are commonly mistaken.


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

```


```python
img_array = mpimg.imread('https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/cat.png')
plt.imshow(img_array)
plt.axis("off")
print (img_array.shape)
```

    (210, 200, 3)

![png](https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/cat.png) <!--notebook-skip-line-->


The color image has the following properties:
    - width: 210 pixels
    - height: 200 pixels
    - colors: 3 (RGB)

Now lets reshape the image in order to exchange width and height dimension.


```python
reshaped = img_array.reshape(200,210,3)
print (reshaped.shape)
plt.imshow(reshaped)
plt.axis("off")
```

    (200, 210, 3)

![png](https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/reshaped_image.png) <!--notebook-skip-line-->


As we can see the first and second dimensions have changed. However the image can't be identified as cat anylonger. In order to understand what happened, let's have a look at the image below.

<img src="https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/reshape.png" style="width:700px;height:300px;">

While the number of rows and columns changed, the layout of the underlying data did not. The pixel values that have been in one row are still in one row. This means for instance that pixel 10 in the upper right corder ends up in the middle of the image instead of the upper left corner. Consequently contextual information gets lost, because the relative position of pixel values is not the same anymore. As one can imagine a neural network would not be able to classify such an image as cat. `Transpose` instead changes both: the dimensions but also the corresponding pixel values.


```python
transposed = img_array.transpose(1,0,2)
plt.imshow(transposed)
plt.axis("off")
```

![png](https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/transposed_image.png) <!--notebook-skip-line-->


As we can see width and height changed, by rotating pixel values by 90 degrees. Transpose does the following:

<img src="https://raw.githubusercontent.com/NRauschmayr/web-data/tutorial_transpose_reshape/mxnet/doc/tutorials/basic/transpose_reshape/transpose.png" style="width:700px;height:300px;">

As shown in the diagram, pixel values that have been in the first row are now in the first column.
## When to transpose with MXNet
In this chapter we discuss when transpose and reshape is used in MXNet. 
#### Channel first for images
Images are usually stored in the format height, wight, channel. When working with convolutional layers, MXNet expects the layout to be `NCHW` (batch, channel, height, width). Consequently, images need to be transposed to have the right format. For instance, you may have a function like the following:
``` 
def transform(data, label): 
     return data.astype(np.float32).transpose((2,0,1))/255.0, float(label)
```
Images may also be stored as 1 dimensional vector, e.g. instead of `[28,28,1]` you may have `[784,1]`. In this situation you need to perform a reshape e.g. `ndarray.reshape((1,28,28))`


#### TNC layout for RNN
When working with LSTM or GRU layers, the default layout for input and ouput tensors has to be `TNC` (sequence length, batch size, and feature dimensions). For instance in the following network the input goes into a 1 dimensional convolution layer and whose output goes into a GRU cell. Here the tensors would mismatch, because `Conv1D` takes data as `NCT`, but GRU  expects it to be `NTC`. 
```
class model(mx.gluon.nn.Block):
    def __init__(self):
        super(model, self).__init__()
        
        self.sequential1 = mx.gluon.nn.Sequential()
        self.sequential2 = mx.gluon.nn.Sequential()
        self.sequential1 = mx.gluon.nn.Sequential()
        self.sequential2 = mx.gluon.nn.Sequential()
        with self.name_scope():    
            self.sequential1.add(mx.gluon.nn.Conv1D(196, kernel_size=15, strides=4)) 
            self.sequential1.add(mx.gluon.nn.BatchNorm(axis=1))
            self.sequential1.add(mx.gluon.nn.Activation(activation='relu')) 
            self.sequential1.add(mx.gluon.nn.Dropout(0.8)) 

            self.sequential2.add(mx.gluon.rnn.GRU(128, layout='NTC'))
```
To ensure that the forward pass does not crash, we need to do a tensor transpose (see below):
```
    def forward(self, X):
        output = self.sequential1(X) 
        return self.sequential2(output.transpose((0,2,1)))
```

#### Check out the Numpy documentation for more details
<https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.reshape.html>

<https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.transpose.html>

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
