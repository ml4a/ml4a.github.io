// how many classes
var NUM_CLASSES = 6;


var samples =[];


// must leave variable name as "sketch"
var sketch = function(s) 
{
  var c = 0;
  var img;

  s.setup = function() {
    s.createCanvas(500, 500);

    img = s.loadImage("assets/guitar.png");

    samples [0] = s.loadSound('http://www.noiseaddicts.com/samples_1w72b820/4928.mp3')
    samples [1] = s.loadSound('assets/F.mp3')
    samples [2] = s.loadSound('assets/G.mp3')
    samples [3] = s.loadSound('assets/Am.mp3')
    samples [4] = s.loadSound('assets/Dm.mp3')
    samples [5] = s.loadSound('assets/Em.mp3')

  };

  s.draw = function() {
    s.background(0);
    s.image(img, 50, 50);
  };

  s.predict = function(predictedClass) {
    if (c != predictedClass) {
      c = predictedClass;
      samples[c].play();  
    }
    
  }

};
