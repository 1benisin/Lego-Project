const Jimp = require('jimp');
const b = require('./_getAllJpgFromFolder');
// import {_getAllJpgFromFolder} from './_getAllJpgFromFolder.js';

const IMAGE_WIDTH = 3024;
const IMAGE_HEIGHT = 4032;

const allImagePaths = b._getAllJpgFromFolder(`${__dirname}/sample-images`);

// console.log(allImagePaths);
for (let i = 0; i < allImagePaths.length; i++) {
  const path = allImagePaths[i];

  const pathComponents = path.split('/');
  const imageName = pathComponents[pathComponents.length - 1];
  // console.log(imageName)

  // process.stdout.write(`${percentComplete}`);
  // process.stdout.clearLine();
  // process.stdout.cursorTo(0);
  // process.stdout.write("\n");

  Jimp.read(path)
    .then((image) => {
      image
        .cover(299, 299)
        // .crop( 0, (IMAGE_HEIGHT-IMAGE_WIDTH)/2, IMAGE_WIDTH, IMAGE_WIDTH ) // crop
        // .resize( 299, 299 )
        .write(`sample-images/processed/${imageName}`); // save
    })
    .catch((err) => {
      console.error(err);
    });
}
