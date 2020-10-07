
// TensorFlow.js for Node,js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const TARGET_CLASSES = require('./ImageNetLabels.js');

const FEATURE_MODEL = "https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/feature_vector/3/default/1"
const fullModelURL = "https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1"
const MODEL_HTTP_URL = 'inception_feature_model/model.json'

const imagePath = "./sample-images/image9.jpg"

const run = async function () {

  // --- BUILD MODEL
  console.log('\r\n Loading model...');
  const inceptionModel = await tf.loadGraphModel('file://' + MODEL_HTTP_URL, { onProgress: (r) => console.log(r) })
  console.log('\r\n Model loaded...');

  console.log('Outputs', inceptionModel.outputs);

  // const dense1 = tf.layers.dense({ units: 32, activation: 'relu' }).apply(inceptionModel.outputs);
  // const dense2 = tf.layers.dense({ units: 10, activation: 'softmax' }).apply(dense1);
  // const myModel = tf.model({ inputs: inceptionModel.input, outputs: dense2 });


  const image = fs.readFileSync(imagePath);
  // const uint8array = new Uint8Array(image);

  const imageTensor = tf.node.decodeImage(image, 3)
    .div(tf.scalar(255))
    .expandDims()
  console.log('input Tensor-----', imageTensor)

  let prediction = inceptionModel.execute(imageTensor)
  // let prediction = inceptionModel.predict(imageTensor)
  console.log('prediction-----', prediction)

  const pdata = await prediction.data();
  let top5 = Array.from(pdata)
  // .map(function (p, i) { // this is Array.map
  //   return {
  //     probability: p,
  //     className: TARGET_CLASSES[i] // we are selecting the value from the obj
  //   };
  // }).sort(function (a, b) {
  //   return b.probability - a.probability;
  // }).slice(0, 10);
  console.log('prediction data-----', top5)


  // console.log('1-----', inceptionModel);
  // console.log('2-----', inceptionModel)

  // console.log('3-----',)
  // console.log('4-----', inceptionModel.inputs)

  // const model = tf.model({ inputs: inceptionModel.inputs, outputs: inceptionModel.outputs });
};


run();