
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

#### Check out the Numpy documentation for more details
<https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.reshape.html>

<https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.transpose.html>

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
