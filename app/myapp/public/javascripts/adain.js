import * as tf from '@tensorflow/tfjs';

export function adain(contentFeatures, styleFeatures) {
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