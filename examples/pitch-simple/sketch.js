let audioContext;
let mic;
let pitch;
let freq;

function setup() {
  createCanvas(500, 500);
  audioContext = getAudioContext();
  mic = new p5.AudioIn();
  mic.start(startPitch);
  freq = 0;
}

function draw() {
  background(0);
  stroke(255);
  noFill();
  var y = map(freq, 0, 2000, height, 0);
  line(0, y, width, y);
}

function startPitch() {
  pitch = ml5.pitchDetection('./model/', audioContext , mic.stream, modelLoaded);
}

function modelLoaded() {
  select('#status').html('Model Loaded');
  getPitch();
}

function getPitch() {
  pitch.getPitch(function(err, frequency) {
    if (frequency) {
      freq = frequency;
      select('#result').html(frequency);
    } else {
      select('#result').html('No pitch detected');
    }
    getPitch();
  });
}

