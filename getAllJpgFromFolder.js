// takes a folder name and return all jpg's including nested in sub folders
const getAllJpgFromFolder = (dir) => {
  const fs = require('fs');
  let results = [];

  fs.readdirSync(dir).forEach((file) => {
    file = `${dir}/${file}`;
    const stat = fs.statSync(file);

    if (stat && stat.isDirectory()) {
      results = results.concat(getAllJpgFromFolder(file));
    } else if (file.match(/jpe?g$/i)) {
      results.push(file);
    }
  });

  return results;
};

// EXAMPLE CALL: _getAllFilesFromFolder(__dirname + "/sample-images")
module.exports = getAllJpgFromFolder;
