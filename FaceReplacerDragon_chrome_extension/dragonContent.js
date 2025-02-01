console.log('Content script loaded');

const newImageURL = "https://i.etsystatic.com/39938216/r/il/430cba/4488972880/il_1588xN.4488972880_3h3k.jpg";

// Function to replace all <img> and <image> elements
async function replaceImages() {
  // Ensure TensorFlow.js is loaded
//   if (typeof tf === 'undefined') {
//     console.error('TensorFlow.js is not loaded');
//     return;
//   }

//   // Load the model
//   const model = await tf.loadLayersModel(chrome.runtime.getURL('./model/face_recognition_model_tfjs/model.json'));

//   // Function to get embedding from image
//   async function getEmbedding(image) {
//     const tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([160, 160]).toFloat().expandDims();
//     const embedding = model.predict(tensor).dataSync();
//     return embedding;
//   }



  // // Replace standard <img> elements
  const imgTags = document.querySelectorAll('img');
  imgTags.forEach(async img => {
    if (img.src !== newImageURL) { // Avoid re-replacing the same image
    //   const isMatch = await isMatchingImage(img);
    //   if (isMatch) {
        console.log('Replacing img tag:', img);
        img.src = newImageURL;
        img.alt = "Replaced Image"; // Optional: Update alt text.
    //   }
    }
  });

  // // Replace <image> elements with xlink:href
  const imageTags = document.querySelectorAll('image');
  imageTags.forEach(async image => {
    const href = image.getAttribute('xlink:href');
    if (href !== newImageURL) { // Avoid re-replacing the same image
    //   const isMatch = await isMatchingImage(image);
    //   if (isMatch) {
        console.log('Replacing <image> tag:', image);
        image.setAttribute('xlink:href', newImageURL);
    //   }
    }
  });

  // Replace specific <image> element with the given xlink:href
  const specificImage = document.querySelector('image[xlink\\:href="https://"]');
  if (specificImage) {
    console.log('Replacing specific <image> tag:', specificImage);
    specificImage.setAttribute('xlink:href', newImageURL);
  }
}

// Replace images on initial load
replaceImages();

// Monitor for dynamically added images
const observer = new MutationObserver(async mutations => {
  for (const mutation of mutations) {
    if (mutation.type === 'childList') {
      mutation.addedNodes.forEach(async node => {
        if (node.tagName === 'IMG') {

            console.log('New img tag added, replacing:', node);
            node.src = newImageURL;
            node.alt = "Replaced Image";
        //   }
        } else if (node.tagName === 'IMAGE') {

            console.log('New <image> tag added, replacing:', node);
            node.setAttribute('xlink:href', newImageURL);
        //   }
        } else if (node.querySelectorAll) {
          // Check for <img> and <image> inside added elements
          node.querySelectorAll('img').forEach(async img => {

              console.log('Replacing newly added img tag:', img);
              img.src = newImageURL;
              img.alt = "Replaced Image";
          });
          node.querySelectorAll('image').forEach(async image => {

              console.log('Replacing newly added <image> tag:', image);
              image.setAttribute('xlink:href', newImageURL);
          });
        }
      });
    }
    else if (mutation.type === 'attributes') {
      if (mutation.attributeName === 'xlink:href') {

          console.log('New <image> tag added, replacing:', mutation.target);
          mutation.target.setAttribute('xlink:href', newImageURL);
      }
    }
  }
});

// Start observing the document for changes
observer.observe(document, {
  childList: true, // Observe direct children
  subtree: true    // Observe all descendants
});

console.log('Observer is monitoring changes');
