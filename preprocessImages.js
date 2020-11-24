const Jimp = require('jimp');
// const getAllJpgFromFolder = require('./getAllJpgFromFolder');

// const DIRECTORY_CONTAINING_IMAGES = `${__dirname}/sample-images/example2`;

const preprocessImages = async (imgPaths) => {
  // console.log(imgPaths);
  for (let i = 0; i < imgPaths.length; i++) {
    const path = imgPaths[i];

    const pathComponents = path.split('/');
    const imageName = pathComponents[pathComponents.length - 1];
    const partName = pathComponents[pathComponents.length - 2];

    const image = await Jimp.read(path);
    image.cover(299, 299).write(`images/processed/${partName}/${imageName}`);
    console.log(((i / imgPaths.length) * 100).toFixed(2), '%');
    // Jimp.read(path)
    //   .then((image) => {
    //     image
    //       .cover(299, 299)
    //       .write(`sample-images/processed/${imageName}`);
    //     console.log(imageName, i);
    //   })
    //   .catch((err) => {
    //     console.error(err);
    //   });
  }
};

module.exports = preprocessImages;
// const allImagePaths = getAllJpgFromFolder(DIRECTORY_CONTAINING_IMAGES);
// console.log('Total images to process: ', allImagePaths.length, ' Images');
// processImages(allImagePaths);
