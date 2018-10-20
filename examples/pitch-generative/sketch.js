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

function draw(){
  if (!freq) {
    return;
  }
  push();
  if (frameCount % 150 == 0) {
    background(255);
  }
  translate(340+400, 400);
  noFill();
  stroke(0, 60);
  beginShape();
  for (var i=0; i<200; i++) {
    var noiseRate = map(freq, 0, 2000, 0.003, 0.075);
    var ang = TWO_PI * (float(i)/200);
    var rad = 600 * noise(i*noiseRate, t*0.005);
    var x = rad * cos(ang);
    var y = rad * sin(ang);
    curveVertex(x, y);
  }
  endShape(CLOSE);
  pop();

  t += 1;
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

