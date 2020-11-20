
// takes a folder name and return all jpg's including nested in sub folders
const _getAllJpgFromFolder = (dir) => {

    const filesystem = require("fs");
    let results = [];

    filesystem.readdirSync(dir).forEach(function(file) {

        file = dir+'/'+file;
        const stat = filesystem.statSync(file);

        if (stat && stat.isDirectory()) {
            results = results.concat(_getAllJpgFromFolder(file))
        } else {
            if(file.match(/jpe?g$/i)){
                results.push(file);
            }
        }

    });

    return results;

};

// EXAMPLE CALL: _getAllFilesFromFolder(__dirname + "/sample-images")
module.exports = {_getAllJpgFromFolder};