const getAllJpgFromFolder = require('./getAllJpgFromFolder');
const preprocessImages = require('./preprocessImages');

const DIRECTORY_CONTAINING_IMAGES = '/Users/benjaminclark/Dropbox/Code/lego_sorter_data/part_color';
// const DIRECTORY_CONTAINING_IMAGES = `${__dirname}/sample-images/example2`;

const allImagePaths = getAllJpgFromFolder(DIRECTORY_CONTAINING_IMAGES);
console.log('Total images to process: ', allImagePaths.length, ' Images');

preprocessImages(allImagePaths);
