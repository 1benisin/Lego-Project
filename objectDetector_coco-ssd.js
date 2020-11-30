const cocoSsd = require('@tensorflow-models/coco-ssd');
const tf = require('@tensorflow/tfjs-node-gpu');
const { privateDecrypt } = require('crypto');
const fs = require('fs').promises;
const Jimp = require('jimp');
const { type } = require('os');

// Load the Coco SSD model and image.
// Promise.all([cocoSsd.load(), fs.readFile('./images/sample/image10.JPG')])
//   .then((results) => {
//   // First result is the COCO-SSD model object.
//     const model = results[0];
//     // Second result is image buffer.
//     const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3);
//     // Call detect() to run inference.
//     return model.detect(imgTensor, 10, 0.1);
//   })
//   .then((predictions) => {
//     // console.log(JSON.stringify(predictions, null, 2));
//     console.log(predictions);
//   });

const run = async () => {
  const imgPath = './images/preprocessed/IMG_0007.JPG';

  // load model
  const model = await cocoSsd.load();

  // get image
  const imgJIMP = await Jimp.read(imgPath);
  const imgbuf = await imgJIMP.getBufferAsync(Jimp.MIME_JPEG);
  const imgTensor = tf.node.decodeImage(new Uint8Array(imgbuf), 3);

  // detect object
  const predictions = await model.detect(imgTensor, 2, 0.01);
  console.log('predictions', predictions);

  // crop image
  const bBox1Area = predictions[0] ? predictions[0].bbox[2] * predictions[0].bbox[2] : null; // area of the 1st predictions bounding box if it exists
  const bBox2Area = predictions[1] ? predictions[1].bbox[2] * predictions[1].bbox[2] : null; // area of the 2nd predictions bounding box if it exists

  const bBox = (!bBox2Area ? predictions[0] : bBox1Area > bBox2Area ? predictions[1] : predictions[0]).bbox; // set to smaller bounding box unless only one prediction exists
  console.log('bBox', bBox);

  const cropBorderLength = (bBox[2] > bBox[3] ? bBox[2] : bBox[3]) * 1.1; // set to longer length and make crop border 10% larger
  const centerXY = {
    x: bBox[0] + bBox[2] / 2,
    y: bBox[1] + bBox[3] / 2,
  };
  console.log('centerXY', centerXY);

  imgJIMP.crop(centerXY.x - cropBorderLength / 2, centerXY.y - cropBorderLength / 2, cropBorderLength, cropBorderLength)
    .write('images/testImage.jpg');

  // resize image
  // save image
};

run();
