console.log(tf.getBackend());

// Primero vendrá el Encoder.

const inputEncoder = tf.input({shape: [(null, null, 3)]});
const preprocess = null; // aqui falta crear la custom layer de preprocesamiento








// Layer de preprocesamiento
 class PreprocessLayer extends tf.layers.Layer {
     constructor() {
         super({});
     }

     computeOutputShape(inputShape) { return inputShape;}

     call(input, kwargs) {
         const mean_pixel = tf.tensor1d([123.68, 116.779, 103.939]);
         const result = tf.mul(tf.scalar(255), input);
         return tf.sub(result, mean_pixel);
     }

     getClassName() { return 'Preprocess'; }
 }

 