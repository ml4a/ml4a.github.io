// how many classes
let numClasses = 3;

let featureExtractor;
let classifier;
let video;
let loss;
let numImages;
let label;
let buttons;

function setup() {
  noCanvas();

  label = 0;
  numImages = Array(numClasses).fill(0);

  video = createCapture(VIDEO);
  video.parent('videoContainer');
  
  featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
  featureExtractor.numClasses = numClasses;
  classifier = featureExtractor.classification(video);
  
  createButtons();
}

function modelReady() {
  select('#loading').html('Base Model (MobileNet) loaded!');
}

function addImage(label) {
  classifier.addImage(label);
}

function classify() {
  console.log("CLASSIFY!")
  classifier.classify(gotResults);
}

function createButtons() {
  buttons = [];
  for (var i=0; i<numClasses; i++) {
    var button = createButton('class '+i);
    button.parent('buttons');
    button.id('class'+i);
    buttons.push(button);
    !function outer(i){
      button.mousePressed(function() {
        addImage(i);
        numImages[i]++;
        select('#class'+i).html('class '+i+': '+numImages[i]);
      });
    }(i)
  }

  // Train Button
  train = select('#train');
  train.mousePressed(function() {
    classifier.train(function(lossValue) {
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
  buttonPredict.mousePressed(classify);
}

function gotResults(err, nextLabel) {
  if (err) {
    console.error(err);
  }
  select('#result').html(nextLabel);
  label = nextLabel;
}

