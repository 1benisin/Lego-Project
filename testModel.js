const tf = require('@tensorflow/tfjs-node');

const csvUrl = 'file://./data/trainData.csv';

const numOfLabels = 48;

async function run() {
  // We want to predict the column "medv", which represents a median value of
  // a home (in $1000s), so we mark it as a label.
  const csvDataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        label: {
          isLabel: true,
        },
      },
    },
  );

  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  // Prepare the Dataset for training.
  const flattenedDataset = csvDataset
    .map(({ xs, ys }) =>
    // Convert xs(features) and ys(labels) from object form (keyed by
    // column name) to array form.
      ({ xs: Object.values(xs), ys: Object.values(ys) }))
    .batch(10);

  // Define the model.
  //   const model = tf.sequential();
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [numOfFeatures], units: 248, activation: 'relu' }),
      tf.layers.dense({ units: 128, activation: 'relu' }),
      //   tf.layers.dropout({ rate: 0.01 }),
      tf.layers.dense({ units: numOfLabels, activation: 'softmax' }),
    ],
  });

  model.summary();

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Fit the model using the prepared Dataset
  return model.fitDataset(flattenedDataset, {
    epochs: 20,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`${epoch}:${logs.loss}`);
      },
    },
  });
}

run();
