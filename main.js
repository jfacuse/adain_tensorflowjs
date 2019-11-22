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
const model = tf.loadLayersModel('file:///home/jorge/IpreStyleTransfer/adain_tensorflowjs/models/content_encoder/model.json');
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




 