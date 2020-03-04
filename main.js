/* eslint-disable require-jsdoc */
import * as tf from '@tensorflow/tfjs';
import '@babel/polyfill';
tf.ENV.set('WEBGL_PACK', false);

class Preprocess extends tf.layers.Layer {
  constructor() {
    super({});
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  call(input, kwargs) {
    let result = tf.reverse(input[0], -1);
    const meanPixel = tf.tensor1d([123.68, 116.779, 103.939]);
    result = tf.mul(tf.scalar(255), result);
    return tf.sub(result, meanPixel);
  }

  static get className() {
    return 'Preprocess';
  }
}
tf.serialization.registerClass(Preprocess);

class Lambda extends tf.layers.Layer {
  constructor() {
    super({});
  }
  call(input, kwargs) {
    // Ojo, el verdadero input viene en input[0], sino da error de Compilar shaders.
    return tf.pad(input[0], [
      [0, 0],
      [1, 1],
      [1, 1],
      [0, 0],
    ]);
  }

  static get className() {
    return 'Lambda';
  }
}
tf.serialization.registerClass(Lambda);

function adain(contentFeatures, styleFeatures) {
  const contentMoments = tf.moments(contentFeatures, [1, 2], true);
  const styleMoments = tf.moments(styleFeatures, [1, 2], true);
  return tf.batchNorm(
    contentFeatures,
    contentMoments.mean,
    contentMoments.variance,
    styleMoments.mean,
    tf.sqrt(styleMoments.variance),
    1e-5
  );
}

function loadImage(element, evt) {
  const file = evt.target.files[0];
  const fileReader = new FileReader();
  fileReader.onload = (e) => {
    document.getElementById(element).src = e.target.result;
  };
  fileReader.readAsDataURL(file);
}

// Initialization of variables
let encoder;
let decoder;
const contentInput = document.getElementById('content');
const styleInput = document.getElementById('style');
const contentImage = document.getElementById('content-img');
const styleImage = document.getElementById('style-img');
const styleSize = document.getElementById('style-img-size');
const stylized = document.getElementById('stylized');
const stylizeButton = document.getElementById('stylize-button');
const styleTransferStatus = document.getElementById('style-transfer-status');
const statusMessage = document.getElementById('status-message');
const styleStrength = document.getElementById('stylization-strength');

function changeStatus(message) {
  statusMessage.textContent = message;
}

contentInput.onchange = (evt) => {
  loadImage('content-img', evt);
};

styleInput.onchange = (evt) => {
  loadImage('style-img', evt);
};

styleSize.oninput = (evt) => {
  styleImage.height = evt.target.value;
};

document.getElementById('stylize-button').onclick = async () => {
  // Desactivar el boton de estilizado primero
  stylizeButton.disabled = true;
  styleTransferStatus.hidden = false;

  if (!encoder || !decoder) {
    changeStatus('Loading encoder');
    encoder = await tf.loadLayersModel(
      'adain_encoder/content_encoder/model.json'
    );
    changeStatus('Loading decoder');
    decoder = await tf.loadLayersModel('adain_decoder/decoder/model.json');
  }

  await tf.nextFrame();
  changeStatus('Generating content representation');

  let bottleneck = await tf.tidy(() => {
    return encoder.predict(
      tf.browser
        .fromPixels(contentImage)
        .toFloat()
        .div(tf.scalar(255))
        .expandDims()
    );
  });

  await tf.nextFrame();
  
  const contentBottleneck = bottleneck;
  changeStatus('Generating style representation');
  bottleneck = await tf.tidy(() => {
    return encoder.predict(
      tf.browser
        .fromPixels(styleImage)
        .toFloat()
        .div(tf.scalar(255))
        .expandDims()
    );
  });

  await tf.nextFrame();
  const styleBottleneck = bottleneck;
  changeStatus('Transfering style using AdaIN');
  await tf.nextFrame();

  let resultAdain = tf.tidy(() => adain(contentBottleneck, styleBottleneck));
  resultAdain = resultAdain.mul(styleStrength.value/100).add(contentBottleneck.mul(1-styleStrength.value/100));

  tf.dispose(contentBottleneck);
  tf.dispose(styleBottleneck);
  changeStatus('Decoding the output');
  await tf.nextFrame();

  bottleneck = await tf.tidy(() => {
    return tf.clipByValue(decoder.predict(resultAdain).squeeze(), 0, 1);
  });

  changeStatus('Drawing the image on the canvas');
  await tf.nextFrame();
  await tf.browser.toPixels(bottleneck, stylized);

  tf.dispose(bottleneck);
  tf.dispose(resultAdain);
  changeStatus('');

  // Una vez terminado, reactivar el boton
  stylizeButton.disabled = false;
  styleTransferStatus.hidden = true;
};
