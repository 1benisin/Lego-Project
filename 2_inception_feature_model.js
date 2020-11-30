// TensorFlow.js for Node,js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const TARGET_CLASSES = require('./ImageNetLabels.js');

const FEATURE_MODEL = 'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/feature_vector/3/default/1';
const fullModelURL = 'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1';
const MODEL_HTTP_URL = 'inception_feature_model/model.json';
const INCEPTION_LAYERS_URL = 'model/inception_model/model.json';

const imagePath = './images/sample/image9.jpg';

const run = async function () {
  // --- BUILD MODEL
  console.log('\r\n Loading model...');
  // const inceptionModel = await tf.loadGraphModel(`file://${MODEL_HTTP_URL}`, { onProgress: (r) => console.log(r) });
  const inceptionModel = await tf.loadLayersModel(`file://${INCEPTION_LAYERS_URL}`, { onProgress: (r) => console.log(r) });
  console.log('\r\n Model loaded...');

  inceptionModel.summary();
  // console.log('Outputs', inceptionModel.outputs);
  // console.log('MODEL', Object.keys(inceptionModel.executor));
  // console.log('MODEL', inceptionModel.executor._inputs[0].rawAttrs.shape.shape.dim);

  // const dense1 = tf.layers.dense({ units: 32, activation: 'relu' }).apply(inceptionModel.outputs);
  // const dense2 = tf.layers.dense({ units: 10, activation: 'softmax' }).apply(dense1);
  // const myModel = tf.model({ inputs: inceptionModel.input, outputs: dense2 });

  const image = fs.readFileSync(imagePath);
  // const uint8array = new Uint8Array(image);

  const imageTensor = tf.node.decodeImage(image, 3)
    .div(tf.scalar(255))
    .expandDims();
  console.log('input Tensor-----', imageTensor);

  // ---------------------------------------------------

  // const input = tf.input({ shape: [5] });
  // const input = tf.input(inceptionModel.inputs[0]);

  // // First dense layer uses relu activation.
  // const denseLayer1 = tf.layers.dense({ units: 10, activation: 'relu' });
  // // Second dense layer uses softmax activation.
  // const denseLayer2 = tf.layers.dense({ units: 4, activation: 'softmax' });

  // // Obtain the output symbolic tensor by applying the layers on the input.
  // const output = denseLayer2.apply(denseLayer1.apply(input));

  // // Create the model based on the inputs.
  // const model = tf.model({ inputs: input, outputs: output });
  // const testModel = tf.model({ inputs: inceptionModel.inputs[0], outputs: inceptionModel.outputs[0] });

  // ---------------------------------------------------

  // const prediction = inceptionModel.execute(imageTensor);
  const prediction = inceptionModel.predict(imageTensor);
  // let prediction = inceptionModel.predict(imageTensor)
  // console.log('prediction-----', prediction);

  console.log('prediction', prediction);
  prediction.print();
  return;

  const pdata = await prediction.data();
  console.log('pdata', pdata);
  const top5 = Array.from(pdata);
  // .map(function (p, i) { // this is Array.map
  //   return {
  //     probability: p,
  //     className: TARGET_CLASSES[i] // we are selecting the value from the obj
  //   };
  // }).sort(function (a, b) {
  //   return b.probability - a.probability;
  // }).slice(0, 10);
  console.log('prediction data-----', top5);

  // console.log('1-----', inceptionModel);
  // console.log('2-----', inceptionModel)

  // console.log('3-----',)
  // console.log('4-----', inceptionModel.inputs)

  // const model = tf.model({ inputs: inceptionModel.inputs, outputs: inceptionModel.outputs });
};

run();
