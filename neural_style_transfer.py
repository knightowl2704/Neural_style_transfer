import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image


from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import backend as K

from PIL import Image
import numpy as np

content_layers = ["block3_conv3"]
style_layers = ["block1_conv1", "block2_conv2", "block4_conv3"]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

content_path = '/media/piyush/Disk/Code practice/Neural_style_transfer/content1.jpg'
style_path = '/media/piyush/Disk/Code practice/Neural_style_transfer/style1.jpg'
save_name = 'try.jpg'

vgg_weights = "/media/piyush/Disk/Code practice/Neural_style_transfer/vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img_size = max(img.size)
    scale = max_dim / img_size
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)


    img = np.expand_dims(img, axis=0)


    out = tf.keras.applications.vgg19.preprocess_input(img)

    return tf.convert_to_tensor(out)


def deprocess_img(processed_img):
    x = processed_img.copy()


    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')

    return x




def get_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target)) / 2


def gram_matrix(input_tensor):
     
        
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]

        
    gram = tf.matmul(a, a, transpose_a=True)

    return gram


def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target)) / (
                channels ** 2 * width * height)


def get_feature_representations(model, content_path, style_path, num_content_layers):
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    content_outputs = model(content_image)
    style_outputs = model(style_image)

    style_features = [style_layer[0] for style_layer in style_outputs[num_content_layers:]]
    content_features = [content_layer[0] for content_layer in content_outputs[:num_content_layers]]

    return style_features, content_features


def compute_loss(model, loss_weights, generated_output_activations, gram_style_features, content_features,
                 num_content_layers, num_style_layers):
    generated_content_activations = generated_output_activations[:num_content_layers]
    generated_style_activations = generated_output_activations[num_content_layers:]

    style_weight, content_weight = loss_weights

    style_score = 0
    content_score = 0


    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, generated_style_activations):
        temp = get_style_loss(comb_style[0], target_style)
        style_score += weight_per_style_layer * temp

    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, generated_content_activations):
        temp = get_content_loss(comb_content[0], target_content)
        content_score += weight_per_content_layer * temp

        
    loss = style_weight * style_score + content_weight * content_score

    return loss, style_score, content_score



    
def get_model(content_layers, style_layers):
        
    vgg19 = VGG19(weights=None, include_top=False)

        
    vgg19.trainable = False

    style_model_outputs = [vgg19.get_layer(name).output for name in style_layers]
    content_model_outputs = [vgg19.get_layer(name).output for name in content_layers]

    model_outputs = content_model_outputs + style_model_outputs

        
    return Model(inputs=vgg19.input, outputs=model_outputs), vgg19


def run_style_transfer(content_path, style_path, num_iterations=100, content_weight=0.1, style_weight=0.9):
        
    sess = tf.Session()

        
    K.set_session(sess)

    model, vgg19 = get_model(content_layers, style_layers)

        
    style_features, content_features = get_feature_representations(model, content_path, style_path, num_content_layers)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

        
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

        
        
        
    generated_image = load_img(content_path)
        

        
    generated_image = tf.Variable(generated_image, dtype=tf.float32)

    model_outputs = model(generated_image)

        
    loss_weights = (style_weight, content_weight)

        
    loss = compute_loss(model, loss_weights, model_outputs, gram_style_features, content_features, num_content_layers,
                        num_style_layers)
    opt = tf.train.AdamOptimizer(learning_rate=9, beta1=0.9, epsilon=1e-1).minimize(loss[0], var_list=[generated_image])

    sess.run(tf.global_variables_initializer())
    sess.run(generated_image.initializer)

        
    vgg19.load_weights(vgg_weights)

        
    best_loss, best_img = float('inf'), None

    for i in range(num_iterations):

            
        sess.run(opt)

            
        clipped = tf.clip_by_value(generated_image, min_vals, max_vals)
            
        generated_image.assign(clipped)

            
        total_loss, style_score, content_score = loss
        total_loss = total_loss.eval(session=sess)

        if total_loss < best_loss:
                
            best_loss = total_loss

                
            temp_generated_image = sess.run(generated_image)[0]
            best_img = deprocess_img(temp_generated_image)

            s_loss = sess.run(style_score)
            c_loss = sess.run(content_score)

                
            print('best: iteration: ', i, 'loss: ', total_loss, '  style_loss: ', s_loss, '  content_loss: ', c_loss)

            
        if (i + 1) % 10 == 0:
            output = Image.fromarray(best_img)
            output.save(str(i + 1) + '-' + save_name)

        
    sess.close()

    return best_img, best_loss


best, best_loss = run_style_transfer(content_path, style_path)