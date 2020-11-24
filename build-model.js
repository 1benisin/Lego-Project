// TensorFlow.js for Node,js
// const tf = require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs-node');

// Fashion-MNIST training & test data
const trainDataUrl = 'file://./data/trainData.csv';
const testDataUrl = 'file://./data/testData.csv';

// mapping of Fashion-MNIST labels (i.e., T-shirt=0, Trouser=1, etc.)
const labels = [
  '2539_dark-gray',
  '30041_dark-bluish-gray',
  '3004_dark-gray',
  '3004_light-grey',
  '3004_red',
  '3004_white',
  '3005_blue',
  '30134_reddish-brown',
  '30169_black',
  '3020_dark-bluish-gray',
  '3022_dark-bluish-gray',
  '3022_dark-orange',
  '30237_dark-grey',
  '3024_black',
  '3024_white',
  '3028_light-gray',
  '30358_dark-bluish-gray',
  '3040_black',
  '3040_light-gray',
  '30541_dark-red',
  '30566_dark-gray',
  '32059_dark-bluish-gray',
  '3403c01_black',
  '3623_red',
  '3660_dark-gray',
  '3665_dark-gray',
  '3666_black',
  '3961pb08_bright-light-orange',
  '4032_brown',
  '40345_dark-gray',
  '4070_red',
  '4070_reddish-brown',
  '4073_trans-clear',
  '4150_light-gray',
  '4151_dark-gray',
  '41764_light-gray',
  '41768_red',
  '41770_reddish-brown',
  '41862_dark-red',
  '43712_red',
  '49668_tan',
  '60481_reddish-brown',
  '6233_chrome-silver',
  '6541_black',
  '87580_dark-bluish-gray',
  '92743pb01_tan',
  '98283_light-gray',
  'non-lego_non-lego',
];
// Build, train a model with a subset of the data
// Use the first n classes
const numOfClasses = labels.length;
const numOfFeatures = 2048;

const batchSize = 32;
const epochsValue = 10;

// load and transform data
const loadData = (dataUrl, batches = batchSize) => {
  // normalize data values between 0-1
  const normalize = ({ xs, ys }) => ({
    xs: Object.values(xs),
    ys: ys.label,
  });

  // transform input array (xs) to 3D tensor
  // binarize output label (ys)
  const transform = ({ xs, ys }) => {
    // array of zeros
    const zeros = (new Array(numOfClasses)).fill(0);

    return {
      xs: tf.tensor1d(Object.values(xs)),
      ys: tf.tensor1d(zeros.map((z, i) => (i === ys.label ? 1 : 0))),
    };
  };

  // load, normalize, transform, batch
  return tf.data
    .csv(dataUrl, { columnConfigs: { label: { isLabel: true } } })
    // .map(normalize)
    // .filter((f) => f.ys < numOfClasses)
    .map(transform)
    .batch(batches)
    .shuffle(5);
};

// Define the model architecture
const buildModel = () => {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [numOfFeatures], units: 128, activation: 'relu' }),
      tf.layers.dense({ units: 128, activation: 'relu' }),
      //   tf.layers.dropout({ rate: 0.01 }),
      tf.layers.dense({ units: numOfClasses, activation: 'softmax' }),
    ],
  });

  //   const model = tf.sequential();

  // add the model layers
  //   model.add(tf.layers.conv2d({
  //     inputShape,
  //     filters: 8,
  //     kernelSize: 5,
  //     padding: 'same',
  //     activation: 'relu',
  //   }));
  //   model.add(tf.layers.maxPooling2d({
  //     poolSize: 2,
  //     strides: 2,
  //   }));
  //   model.add(tf.layers.conv2d({
  //     filters: 16,
  //     kernelSize: 5,
  //     padding: 'same',
  //     activation: 'relu',
  //   }));
  //   model.add(tf.layers.maxPooling2d({
  //     poolSize: 3,
  //     strides: 3,
  //   }));
  //   model.add(tf.layers.flatten());
  //   model.add(tf.layers.dense({
  //     units: numOfClasses,
  //     activation: 'softmax',
  //   }));

  //   layers.Dense(128, activation='relu'),
  //   layers.Dense(128, activation='relu'),
  //   layers.Dropout(.1),
  //   layers.Dense(1)

  // compile the model
  //   const optimizer = tf.train.adam(0.05);
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// train the model against the training data
const trainModel = async (model, trainingData, epochs = epochsValue) => {
  const options = {
    epochs,
    // verbose: 1,
    callbacks: {
      onEpochBegin: async (epoch, logs) => {
        // console.log(`Epoch ${epoch + 1} of ${epochs} ...`);
      },
      onEpochEnd: async (epoch, logs) => {
        // console.log(`  train-set loss: ${logs.loss.toFixed(4)}`);
        // console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`);
      },
    },
  };

  return model.fitDataset(trainingData, options);
};

// verify the model against the test data
const evaluateModel = async (model, testingData) => {
  const result = await model.evaluateDataset(testingData);
  const testLoss = result[0].dataSync()[0];
  const testAcc = result[1].dataSync()[0];

  console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
  console.log(`  test-set accuracy: ${testAcc.toFixed(4)}`);
};

// run
const run = async () => {
  const trainData = loadData(trainDataUrl);
  const testData = loadData(testDataUrl);

  //   await testData.forEachAsync((e) => console.log(e));
  const arr = await trainData.take(1).toArray();
  arr[0].ys.print();
  arr[0].xs.print();

  // Full path to the directory to save the model in
  const saveModelPath = 'file://./data';

  const model = buildModel();
  model.summary();

  const info = await trainModel(model, trainData);
  console.log('\r\n', info);
  console.log('\r\nEvaluating model...');
  await evaluateModel(model, testData);
  console.log('\r\nSaving model...');
  await model.save(saveModelPath);
};

run();
// console.log('RESULTS', loadData());
