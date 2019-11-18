tf.ENV.set('WEBGL_PACK', false);
console.log(tf.getBackend());

//tf.enableDebugMode();

// Layer de preprocesamiento
class PreprocessLayer extends tf.layers.Layer {
    constructor() {
        super({});
    }

    computeOutputShape(inputShape) { return inputShape; }

    call(input, kwargs) {
        const mean_pixel = tf.tensor1d([123.68, 116.779, 103.939]);
        const result = tf.mul(tf.scalar(255), input[0]);
        return tf.sub(result, mean_pixel);
    }

    getClassName() { return 'Preprocess'; }
}
// Primero vendrá el Encoder.
// Las layers estan en orden
const encoder = tf.tidy(()=>{
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

return tf.model({
    inputs: inputEncoder,
    outputs: output
});
});

// Ahora viene el DECODER
const decoder = tf.tidy(() => {
const inputDecoder = tf.input({shape: [null, null, 512]});

const deConv2d_4_1 = tf.layers.conv2d({
  filters: 256,
  kernelSize: 3,
  activation: 'relu',
  padding: 'valid',
  name: 'deConv4_1',
  trainable: false
});
const upSampling3 = tf.layers.upSampling2d({
    name: 'upSampling3',
    trainable: false
});
const deConv2d_3_4 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv3_4',
    trainable: false
});
const deConv2d_3_3 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv3_3',
    trainable: false
});
const deConv2d_3_2 = tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv3_2',
    trainable: false
});
const deConv2d_3_1 = tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv3_1',
    trainable: false
});
const upSampling2 = tf.layers.upSampling2d({
    name: 'upSampling2',
    trainable: false,
});
const deConv2d_2_2 = tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv2_2',
    trainable: false
});
const deConv2d_2_1 = tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv2_1',
    trainable: false
});
const upSampling1 = tf.layers.upSampling2d({
    name: 'upSampling1',
    trainable: false,
});
const deConv2d_1_2 = tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'valid',
    name: 'deConv1_2',
    trainable: false
});
const deConv2d_1_1 = tf.layers.conv2d({
    filters: 3,
    kernelSize: 3,
    padding: 'valid',
    name: 'deConv1_1',
    trainable: false
});


let outputDecoder = deConv2d_4_1.apply(inputDecoder);
outputDecoder = upSampling3.apply(outputDecoder);
outputDecoder = deConv2d_3_4.apply(outputDecoder);
outputDecoder = deConv2d_3_3.apply(outputDecoder);
outputDecoder = deConv2d_3_2.apply(outputDecoder);
outputDecoder = deConv2d_3_1.apply(outputDecoder);
outputDecoder = upSampling2.apply(outputDecoder);
outputDecoder = deConv2d_2_2.apply(outputDecoder);
outputDecoder = deConv2d_2_1.apply(outputDecoder);
outputDecoder = upSampling1.apply(outputDecoder);
outputDecoder = deConv2d_1_2.apply(outputDecoder);
outputDecoder = deConv2d_1_1.apply(outputDecoder);

return tf.model({
    inputs: inputDecoder,
    outputs: outputDecoder
});
});

function adain(contentFeatures, styleFeatures) {
    const contentMoments = tf.momments(contentFeatures, [1,2], true);
    const styleMoments = tf.momments(styleFeatures, [1,2], true);
    return tf.batchNorm(
        contentFeatures,
        contentMoments.mean,
        contentMoments.variance,
        styleMoments.mean,
        tf.sqrt(styleMoments.variance),
        1e-5)
}

// Desde aqui pasa a ser las funciones para la vista.
const contentImg = document.getElementById('content');
const styleImg = document.getElementById('style');
function loadImage(event, imgElement) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imgElement.src = e.target.result;
    };
    reader.readAsDataURL(event.target.files[0]);
  }

function loadContent(event) {
    loadImage(event, contentImg);
}

function loadStyle(event) {
    loadImage(event, styleImg);
}

async function styleTransfer(content, style, alpha=1) {
    const contentEncoded = await tf.tidy(() => {
        return encoder.predict(tf.browser.fromPixels(content).toFloat().div(tf.scalar(255)).expandDims());
    });
    //const styleEncoded = await tf.tidy(() => {
    //    return encoder.predict(tf.browser.fromPixels(style).toFloat().div(tf.scalar(255)).expandDims());
    //});
    console.log(tf.memory());
    //const adainResult = adain(contentEncoded, styleEncoded) * alpha + (1-alpha) * contentEncoded;
    //const decoded = decoder.predict(adainResult);
    //return decoded
}

async function stylize(alpha=1) {
    const stylized =  await styleTransfer(contentImg, styleImg, alpha);
    console.log(stylized);
}




 