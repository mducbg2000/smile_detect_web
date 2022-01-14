// import * as tf from "@tensorflow/tfjs";
// import * as faceapi from "face-api.js";

// lấy phần tử video từ dom
const video = document.getElementsByTagName("video")[0];
const btn = document.getElementsByTagName("button")[0];

// load face detector from public/models folder
faceapi.nets.tinyFaceDetector.loadFromUri("/models").then(startVideo);
// load smile detector model from public/models folder
const model = await tf.loadLayersModel("/models/model.json");

// start webcam and assign webcam stream to video element
function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => {
      video.srcObject = stream;
      // when video play, process video
      video.play().then(processVideo);
    },
    (err) => console.error(err)
  );
}

// toggle video play/pause by click to any position in document body
document.body.addEventListener("click", () => {
  video.paused ? startVideo() : video.pause();
});


const processVideo = () => {
  // create a canvas from video
  const canvas = faceapi.createCanvasFromMedia(video);
  // append canvas to body
  document.body.append(canvas);
  // adjusted width and height
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  // loop a frame from video each 500ms
  const loopVideo = setInterval(async () => {
    // create face detection
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions()
    );
    
    // extract the image regions that contain faces
    const faceCanvases = await faceapi.extractFaces(video, detections);
    
    // label for each face, smile or not smile
    let labels = [];
  
    // process each face detected
    faceCanvases.forEach((faceCanvas) => {

      // convert face canvas to grayscale
      let tensor = rgbToGray(faceCanvas);
      // convert to float type
      tensor = tensor.cast("float32").div(255);
      // resize to [32, 32]
      tensor = tensor.resizeBilinear([32, 32]);

      tensor = tf.expandDims(tensor, null);
      // predict the face
      let prediction = model.predict(tensor).arraySync();
      // this 2 value is percentage
      let [not_smile, smile] = prediction[0];
      let label = smile > not_smile ? "Smile" : "Not smile";
      labels.push[label];
      if (smile > not_smile) console.log("Smile: " + smile);
    });
  
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);
    if (video.paused) clearInterval(loopVideo);
  }, 500);
};



function rgbToGray(canvas) {
  let image = tf.browser.fromPixels(canvas, 3);
  const rgb_weights = [0.2989, 0.587, 0.114];
  image = tf.mul(image, rgb_weights);
  image = tf.sum(image, -1);
  image = tf.expandDims(image, -1);
  return image;
}

/**
 *
 * @param {HTMLCanvasElement} canvas
 * @returns new canvas with image data from old canvas
 */
function extractFrame(canvas) {
  const frame = document.createElement("canvas");
  frame.getContext("2d").drawImage(canvas, 0, 0);
  return frame;
}
