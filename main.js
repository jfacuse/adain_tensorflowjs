console.log(tf.getBackend());

// Layer de preprocesamiento
class PreprocessLayer extends tf.layers.Layer {
    constructor() {
        super({});
    }

    computeOutputShape(inputShape) { return inputShape; }

    call(input, kwargs) {
        const mean_pixel = tf.tensor1d([123.68, 116.779, 103.939]);
        const result = tf.mul(tf.scalar(255), input);
        return tf.sub(result, mean_pixel);
    }

    getClassName() { return 'Preprocess'; }
}
// Primero vendrá el Encoder.
// Las layers estan en orden

const inputEncoder = tf.input({shape: [null, null, 3]});
const preprocess = new PreprocessLayer();
const conv2d_1_1 = tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block1_conv1',
    trainable: false
});

const conv2d_1_2 = tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block1_conv2',
    trainable: false
});

const maxPooling_1 = tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
    name: 'block1_pool',
    trainable: false
});

const conv2d_2_1 = tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block2_conv1',
    trainable: false
});

const conv2d_2_2 = tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block2_conv2',
    trainable: false
});

const maxPooling_2 = tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
    name: 'block2_pool',
    trainable: false
});

const conv2d_3_1 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block3_conv1',
    trainable: false
});

const conv2d_3_2 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block3_conv2',
    trainable: false
});

const conv2d_3_3 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block3_conv3',
    trainable: false
});

const conv2d_3_4 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block3_conv4',
    trainable: false
});

const maxPooling_3 = tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
    name: 'block3_pool',
    trainable: false
});

const conv2d_4_1 = tf.layers.conv2d({
    filters: 512,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'block4_conv1',
    trainable: false
});

let output = preprocess.apply(inputEncoder);
console.log(output);
output = conv2d_1_1.apply(output);
output = conv2d_1_2.apply(output);
output = maxPooling_1.apply(output);
output = conv2d_2_1.apply(output);
output = conv2d_2_2.apply(output);
output = maxPooling_2.apply(output);
output = conv2d_3_1.apply(output);
output = conv2d_3_2.apply(output);
output = conv2d_3_3.apply(output);
output = conv2d_3_4.apply(output);
output = maxPooling_3.apply(output);
output = conv2d_4_1.apply(output);

const encoder = tf.model({
    inputs: inputEncoder,
    outputs: output
});







 