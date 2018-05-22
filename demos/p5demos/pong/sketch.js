// how many classes
var NUM_CLASSES = 2;

// must leave variable name as "sketch"
var sketch = function(s) 
{
  var c = 0;
  

  var k = 3; //k can be any integer
  
  var pixelColors = [];
  
  //PONG
  var ballX = 400,
    ballY = 240,
    ballSpeedX = 20,
    ballSpeedY = 12;

  var ballSize = 50;

  var player1Score = 0,
    player2Score = 0;

  var WINNING_SCORE = 3;

  var showingWinScreen = false;

  var paddle1Y,
    paddle2Y;
  var PADDLE_HEIGHT = 200;
  var PADDLE_THICKNESS = 25;

  var pongWidth = 800;

  var playGame = false;
  var test = -1;



  s.setup = function() {
    s.createCanvas(800, 480);

    paddle1Y = s.height / 2 - PADDLE_HEIGHT / 2;
    paddle2Y = s.height / 2 - PADDLE_HEIGHT / 2;


  };

  s.draw = function() {
    //s.background("#E4C000");
    s.background(0,0,0);



    //PONG
    s.push();
    s.noStroke();
    if(playGame) s.moveEverything();
    s.drawEverything();
    s.pop();

  };





  s.moveEverything = function() {

    if (showingWinScreen) return;

    ballX += ballSpeedX;
    ballY += ballSpeedY;

    var paddleSpeed = 5;

    if (test>-1) {
      //if (keyIsPressed) {
      if (test == 0) {
        paddle1Y += paddleSpeed;
        //} else if (mouseIsPressed) {
      } else if (test == 1) {
        paddle1Y -= paddleSpeed;
      }
    }

    paddle1Y = s.constrain(paddle1Y, 0, s.height - PADDLE_HEIGHT);


    s.computerMovement();

    var damp = 0.2;

    if (ballX < (0 + ballSize / 2 + PADDLE_THICKNESS)) {
      if (ballY > paddle1Y && ballY < paddle1Y + PADDLE_HEIGHT) {
        ballSpeedX *= -1;
        var deltaY = ballY - (paddle1Y + PADDLE_HEIGHT / 2);
        ballSpeedY = deltaY * damp;
      } else {
        player2Score++; // must be BEFORE ballReset()
        s.ballReset();
      }
    }

    if (ballX > (pongWidth - ballSize / 2 - PADDLE_THICKNESS)) {
      if (ballY > paddle2Y && ballY < paddle2Y + PADDLE_HEIGHT) {
        ballSpeedX *= -1;
        var deltaY = ballY - (paddle2Y + PADDLE_HEIGHT / 2);
        ballSpeedY = deltaY * damp;
      } else {
        player1Score++; // must be BEFORE ballReset()
        s.ballReset();
      }
    }

    if (ballY > s.height) {
      ballY = s.height;
      ballSpeedY *= -1;
    }

    if (ballY < 0) {
      ballY = 0;
      ballSpeedY *= -1;
    }
    ballSpeedY = s.constrain (ballSpeedY, -5,5);
  }

  s.computerMovement = function() {
    var paddle2YCenter = paddle2Y + PADDLE_HEIGHT / 2;
    if (paddle2YCenter < ballY - PADDLE_HEIGHT / 3) {
      paddle2Y += 6;
    } else if (paddle2YCenter > ballY + PADDLE_HEIGHT / 3) {
      paddle2Y -= 6;
    }
  }

  s.ballReset = function() {

    if (player1Score >= WINNING_SCORE || player2Score >= WINNING_SCORE) {
      showingWinScreen = true;
    }

    ballSpeedX *= -1;
    ballX = pongWidth / 2;
    ballY = s.height / 2;

  }

  s.drawEverything = function() {
    s.fill(255,219,0);
    s.rect(0,0,PADDLE_THICKNESS,s.height);
    
    s.fill(255);
    s.noStroke();

    if (showingWinScreen) {
      s.textSize(20);
      if (player1Score >= WINNING_SCORE) {
        s.text("left player won!", 200, 200);
      } else if (player2Score >= WINNING_SCORE) {
        s.text("right player won!", pongWidth - 200, 200);
      }
      s.textSize(14);
      s.text("click to continue", pongWidth / 2, s.height - 200);
      return;
    }

    s.fill("#00B552");

    for (var i = 0; i < s.height; i += 40) {
      s.rect(pongWidth / 2 - 1, i, 2, 20);
    }
    s.rect(0, paddle1Y, PADDLE_THICKNESS, PADDLE_HEIGHT); // left paddle
    s.rect(pongWidth - PADDLE_THICKNESS, paddle2Y, PADDLE_THICKNESS, PADDLE_HEIGHT); // right paddle

    s.fill("FAFBDF");
    s.ellipse(ballX, ballY, ballSize, ballSize); // ball //15 

    s.textSize(20);
    s.text(player1Score, 200, 100);
    s.text(player2Score, pongWidth - 200, 100);
  }





  s.predict = function(predictedClass) {
    c = predictedClass;
    test = c;
  }

  s.keyPressed = function() {
    if (s.key == ' '){
      playGame = true;
    }
  }

};
