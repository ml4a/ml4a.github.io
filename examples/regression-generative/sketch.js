let featureExtractor;
let regressor;
let video;
let loss;
let slider;
let addSample;
let samples = 0;
let positionX = 140;
let t = 0;

function setup() {
  createCanvas(1024, 768);
  video = createCapture(VIDEO);
  video.hide();
  featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
  regressor = featureExtractor.regression(video);
  createButtons();
}

function draw() {
  background(255);
  for (var i=0; i<width; i++) {
    stroke(255 * noise(0.025 * i, t));
    line(i, 0, i, height);
  }
  t += (0.1 * slider.value());
  console.log(t);
  image(video, 0, 0, 240, 180);
}

// A function to be called when the model has been loaded
function modelReady() {
  select('#loading').html('Model loaded!');
}

// Classify the current frame.
function predict() {
  regressor.predict(gotResults);
}

// A util function to create UI buttons
function createButtons() {
  slider = select('#slider');
  // When the Dog button is pressed, add the current frame
  // from the video with a label of "dog" to the classifier
  addSample = select('#addSample');
  addSample.mousePressed(function() {
    regressor.addImage(slider.value());
    select('#amountOfSamples').html(samples++);
  });

  // Train Button
  train = select('#train');
  train.mousePressed(function() {
    regressor.train(function(lossValue) {
      if (lossValue) {
        loss = lossValue;
        select('#loss').html('Loss: ' + loss);
      } else {
        select('#loss').html('Done Training! Final Loss: ' + loss);
      }
    });
  });

  // Predict Button
  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(predict);
}

// Show the results
function gotResults(err, result) {
  if (err) {
    console.error(err);
  }
  positionX = map(result, 0, 1, 0, width);
  slider.value(result);
  predict();
}

