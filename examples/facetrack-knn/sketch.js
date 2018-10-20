var k = 5; //k can be any integer
var machine = new kNear(k);

var currentClass = 0;

var capture;
var tracker
var w = 640 * 1.5,
  h = 480 * 1.5;

var test;

var nSamples = 0;

var normalized = [];

function setup() {
  capture = createCapture(VIDEO);
  createCanvas(w, h);
  capture.size(w, h);
  capture.hide();

  tracker = new clm.tracker();
  tracker.init(pModel);
  tracker.start(capture.elt);
}

function draw() {
  //background(0);
  image(capture, 0, 0, w, h);
  var positions = tracker.getCurrentPosition();

  noFill();
  stroke(255);


  if (positions.length > 1) { //needed to avoid error if no face is found.
    
    //get the bounding box
    var minX = width;
    var minY = height;
    var maxX = 0;
    var maxY = 0;
    
    for (var i = 0; i < positions.length; i++) {
     if (positions[i][0] < minX) {
       minX = positions[i][0];
     } if (positions[i][1] < minY) {
       minY = positions[i][1];
     } if (positions[i][0] > maxX) {
       maxX = positions[i][0];
     } if (positions[i][1] > maxY) {
       maxY = positions[i][1];
     }
    }
  
    //rect(minX,minY, maxX-minX, maxY-minY);
    
    normalized=[];
    
    for (var i = 0; i < positions.length; i++) {
      normalized.push((positions[i][0] - minX) / (maxX-minX));
      normalized.push((positions[i][1] - minY) / (maxY-minY));
      //text("x: " + nf(normalized[i],1,2) + "\n" + "y: " + nf(normalized[i],1,2), positions[i][0], positions[i][1]);
    }
   
  //Draw the face outline
    //Chin outline
    beginShape();
    curveVertex(positions[0][0], positions[1][1]);
    for (var i = 0; i < 16; i++) {
      curveVertex(positions[i][0], positions[i][1]);
    }
    endShape();


    //Right eyebrow curves
    beginShape();
    curveVertex(positions[15][0], positions[15][1]);
    for (var i = 15; i < 19; i++) {
      curveVertex(positions[i][0], positions[i][1]);
      //vertex(positions[i][0], positions[i][1]);
    }
    curveVertex(positions[18][0], positions[18][1]);
    endShape();


    //Left eyebrow curves
    beginShape();
    curveVertex(positions[19][0], positions[19][1]);
    for (var i = 19; i < 23; i++) {
      curveVertex(positions[i][0], positions[i][1]);
      //vertex(positions[i][0], positions[i][1]);
    }
    curveVertex(positions[22][0], positions[22][1]);
    endShape();


    //Left eye
    beginShape();
    curveVertex(positions[23][0], positions[23][1]);
    for (var i = 23; i < 27; i++) {
      curveVertex(positions[i][0], positions[i][1]);
    }
    endShape(CLOSE);

    //Right eye
    beginShape();
    curveVertex(positions[28][0], positions[28][1]);
    for (var i = 28; i < 32; i++) {
      curveVertex(positions[i][0], positions[i][1]);
    }
    endShape(CLOSE);


    //Outer mouth
    beginShape();
    curveVertex(positions[44][0], positions[44][1]);
    for (var i = 44; i < 56; i++) {
      curveVertex(positions[i][0], positions[i][1]);
    }
    endShape(CLOSE);

    //Inner mouth
    beginShape();
    curveVertex(positions[56][0], positions[56][1]);
    for (var i = 56; i < 62; i++) {
      curveVertex(positions[i][0], positions[i][1]);
    }
    endShape(CLOSE);

    beginShape();
    curveVertex(positions[34][0], positions[34][1]);
    for (var i = 34; i < 42; i++) {
      curveVertex(positions[i][0], positions[i][1]);
      if (i == 36) curveVertex(positions[42][0], positions[42][1]); // left nose strill
      if (i == 37) curveVertex(positions[43][0], positions[43][1]); // right nose strill
    }
    endShape();

    //NoseBone
    beginShape();
    vertex(positions[33][0], positions[33][1]);
    vertex(positions[41][0], positions[41][1]);
    vertex(positions[62][0], positions[62][1]);

    endShape(CLOSE);
  }

  if (mouseIsPressed) {
    machine.learn(normalized, currentClass);
    nSamples++;
    
  fill(255, 0, 0);
  noStroke();
  ellipse(width - 25, 25, 25, 25);
    
  } else if (nSamples >0)  {
  fill(0,255,0);
  test = machine.classify(normalized);
  textSize(126);
  text(test, width/2, 3*(height/4));
  
  }

  //test = machine.classify(normalized); 
  
  noStroke();
  fill(0);
  textSize(24);
  text("press [0-9] to change current class --- hold mouse to record samples", 10, 35);
  textSize(36);
    text("trainingClass: " + currentClass, 10, 75);
  text(" nSamples: " + nSamples, width -350, 75);
  //text("trainingClass: " + currentClass + "   prediction: " + test + "   nSamples: " + nSamples, 10, 75);
  
}

function keyPressed() {
  if (key == '0') {
    currentClass = 0;
  } else if (key == '1') {
    currentClass = 1;
  } else if (key == '2') {
    currentClass = 2;
  } else if (key == '3') {
    currentClass = 3;
  } else if (key == '4') {
    currentClass = 4;
  } else if (key == '5') {
    currentClass = 5;
  } else if (key == '6') {
    currentClass = 6;
  } else if (key == '7') {
    currentClass = 7;
  } else if (key == '8') {
    currentClass = 8;
  } else if (key == '9') {
    currentClass = 9;
  }
}