# Neural Style Transfer

**Tensorflow and Keras.**

The model architecture : **VGG19**

**Neural Style Transfer** employs a pretrained convolution neural network (CNN) to transfer styles from a given image to another. The major insight that opens up in this approach is that, "it is possible to separate the style representation and content representations in a CNN, learnt during a computer vision task"

So what we are doing here is we are transfering styles from one image to other. This can be done by defining a loss function that minimizes the activation difference between the content image and the style image.

<h2> Architecture </h2>
<img src = "https://miro.medium.com/max/809/1*ZgW520SZr1QkGoFd3xqYMw.jpeg", width = "512">

The content image is the one on which the style is to be transferred. 

The style image is the one whose style is to be transferred.

The generated image is the image that contains the final result.

<h2> Loss Function </h2>

We define two loss functions: the content loss function and the style loss function. The content loss function ensures that the activations of the higher layers are similar between the content image and the generated image. The style loss function makes sure that the correlation of activations in all the layers are similar between the style image and the generated image.

The final loss is defined as,

<img src="https://miro.medium.com/max/505/1*w7VAfUKbRYG2KXIFBUhfjQ.png">

Content Image 

<img src="https://github.com/knightowl2704/Neural_style_transfer/blob/master/content.jpg" width="512"> 

Style Image

<img src="https://github.com/knightowl2704/Neural_style_transfer/blob/master/style.jpg" width="512"> 

Results :


After 10 iterations ; learning rate of 9 

<img src="https://github.com/knightowl2704/Neural_style_transfer/blob/master/10-generated.jpg" width="512"> 




After 200 iterations ; learning rate of 9

<img src="https://github.com/knightowl2704/Neural_style_transfer/blob/master/200-generated.jpg" width="512"> 



