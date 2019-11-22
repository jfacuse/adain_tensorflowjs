//const tf = require('@tensorflow/tfjs-node-gpu');
tf.ENV.set('WEBGL_PACK', false);
tf.enableDebugMode();

class Lambda extends tf.layers.Layer {
    constructor() {
        super({});
    }

    computeOutputShape(inputShape) { return inputShape; }

    call(input, kwargs) {
        const mean_pixel = tf.tensor1d([123.68, 116.779, 103.939]);
        const result = tf.mul(tf.scalar(255), input[0]);
        return tf.sub(result, mean_pixel);
    }

    static get className() { return 'Lambda'; }
}
tf.serialization.registerClass(Lambda);
const contentImg = document.getElementById('content');
const styleImg = document.getElementById('style');
let encoder;
async function setup() {
    encoder = await tf.loadLayersModel('../models/content_encoder/model.json')
}
setup();
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
    //console.log(encoder);
    const contentEncoded = await tf.tidy(() => {
        return encoder.predict(tf.browser.fromPixels(content).toFloat().div(tf.scalar(255)).expandDims());
    });
    //const styleEncoded = await tf.tidy(() => {
    //    return encoder.predict(tf.browser.fromPixels(style).toFloat().div(tf.scalar(255)).expandDims());
    //});
    //console.log(tf.memory());
    //const adainResult = adain(contentEncoded, styleEncoded) * alpha + (1-alpha) * contentEncoded;
    //const decoded = decoder.predict(adainResult);
    //return decoded
    //return adainResult;
}

async function stylize(alpha=1) {
    const stylized =  await styleTransfer(contentImg, styleImg, alpha);
    console.log(stylized);
}