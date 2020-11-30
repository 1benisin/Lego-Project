// TensorFlow.js for Node,js
const automl = require('@tensorflow/tfjs-automl');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const createCsvWriter = require('csv-writer').createArrayCsvWriter;
const TARGET_CLASSES = require('./ImageNetLabels.js');
const getAllJpgFromFolder = require('./getAllJpgFromFolder');

const DIRECTORY_CONTAINING_IMAGES = `${__dirname}/images/processed`;
const MODEL_PATH = 'model/autoML_model/model.json';
const TEST_IMAGE_PATH = './images/preprocessed/IMG_0007.JPG';

const objectDetect = async () => {
  // --- BUILD MODEL

  const modelUrl = `file://${MODEL_PATH}`; // URL to the model.json file.
  // const model = await automl.loadObjectDetection(modelUrl);
  const model = await tf.loadGraphModel(`file://${MODEL_PATH}`, { onProgress: (r) => console.log(r) });
  console.log(model);
  const image = fs.readFileSync(TEST_IMAGE_PATH);
  const inputImageTensor = tf.node.decodeImage(image, 3);
  const predictions = await model.predict(inputImageTensor);
  console.log(predictions);

  // const model = await tf.loadGraphModel(`file://${MODEL_PATH}`, { onProgress: (r) => console.log(r) });
  // console.log('\r\n Model loaded...');
  // console.log('\r\n model', model.inputs);
  // return;
  // const image = fs.readFileSync(TEST_IMAGE_PATH);
  // const inputImageTensor = tf.node.decodeImage(image, 3)
  //   .div(tf.scalar(255))
  //   .expandDims();

  // const outputTensor = model.execute(inputImageTensor);
  // const outputData = await outputTensor.data();
  // console.log('outputData', outputData);
  // const features = Array.from(outputData);

  // return;

  // // --- GET ALL IMAGE PATHS
  // const imgPaths = getAllJpgFromFolder(DIRECTORY_CONTAINING_IMAGES);
  // console.log('Total images to process: ', imgPaths.length, ' Images');

  // // --- CREATE CSV FILE
  // const csvWriter = createCsvWriter({
  //   path: 'data/image_features.csv',
  //   append: true,
  // });

  // // --- EXTRACT FEATURES FOR EACH IMAGE
  // for (let i = 0; i < imgPaths.length; i++) {
  //   const path = imgPaths[i];
  //   const pathComponents = path.split('/');
  //   const partName = pathComponents[pathComponents.length - 2];

  //   const image = fs.readFileSync(path);

  //   const inputImageTensor = tf.node.decodeImage(image, 3)
  //     .div(tf.scalar(255))
  //     .expandDims();

  //   const outputTensor = inceptionModel.execute(inputImageTensor);
  //   const outputData = await outputTensor.data();
  //   const features = Array.from(outputData);

  //   await csvWriter.writeRecords([[partName, ...features]]);
  // }
};

objectDetect();
