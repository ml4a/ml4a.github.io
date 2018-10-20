// how many classes
let numClasses = 6;

let featureExtractor;
let classifier;
let video;
let loss;
let img;
let numImages;
let samples;
let label;
let buttons;

function setup() {
  createCanvas(500, 500);

  img = loadImage("./assets/guitar.png");

  samples = [];
  samples[0] = loadSound('./assets/C.mp3')
  samples[1] = loadSound('./assets/F.mp3')
  samples[2] = loadSound('./assets/G.mp3')
  samples[3] = loadSound('./assets/Am.mp3')
  samples[4] = loadSound('./assets/Dm.mp3')
  samples[5] = loadSound('./assets/Em.mp3')

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
  if (label != nextLabel) {
    label = nextLabel;
    samples[int(label)].play();  
  }
  classify();
}

function draw() {
  background(0);
  image(img, 50, 50);
}
