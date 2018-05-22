// how many classes
var NUM_CLASSES = 3;

// must leave variable name as "sketch"
var sketch = function(s) 
{
  var c = 0;

  s.setup = function() {
    s.createCanvas(500, 500);
  };

  s.draw = function() {
    s.background(0);
    
    if (c == 0) {
      s.fill(255,0,0);
    } else if (c == 1) {
      s.fill(0,255,0);
    } else if (c == 1) {
      s.fill(0,0,255);
    }
    
    s.rect(150, 150, 200, 200);
  };

  s.predict = function(predictedClass) {
    c = predictedClass;
  }

};
