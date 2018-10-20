let pongWidth = 800;
let pongHeight = 560;
let WINNING_SCORE = 3;
let PADDLE_HEIGHT = 200;
let PADDLE_THICKNESS = 25;
let paddleSpeed = 5;
let ballSpeedX = 10;
let ballSpeedY = 6;
let ballSize = 50;

let featureExtractor;
let regressor;
let video;
let loss;
let slider;
let addSample;
let samples = 0;

let ballX = pongWidth/2,
  ballY = pongHeight/2;
let player1Score = 0,
  player2Score = 0;
let showingWinScreen = false;
let playGame = false;
let paddle1Y, paddle2Y;


function setup() {
  createCanvas(pongWidth, pongHeight);
  
  video = createCapture(VIDEO);
  video.hide();
  
  featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
  regressor = featureExtractor.regression(video);
  
  createButtons();

  paddle1Y = height / 2 - PADDLE_HEIGHT / 2;
  paddle2Y = height / 2 - PADDLE_HEIGHT / 2;
}

function draw() {
  background("#E4C000");
  
  if(playGame) {
    moveEverything();
  } 

  drawEverything();

  if(!playGame) {
    image(video, 0, 0, 320, 240);
  }
}

function updatePaddle(newPaddleValue) {
  paddle1Y = paddle1Y = lerp(paddle1Y, pongHeight * newPaddleValue, 0.2);
  paddle1Y = constrain(paddle1Y, 0, height - PADDLE_HEIGHT);
}

function moveEverything() {
  if (showingWinScreen) return;

  ballX += ballSpeedX;
  ballY += ballSpeedY;

  computerMovement();

  var damp = 0.2;

  if (ballX < (0 + ballSize / 2 + PADDLE_THICKNESS)) {
    if (ballY > paddle1Y && ballY < paddle1Y + PADDLE_HEIGHT) {
      ballSpeedX *= -1;
      var deltaY = ballY - (paddle1Y + PADDLE_HEIGHT / 2);
      ballSpeedY = deltaY * damp;
    } else {
      player2Score++; // must be BEFORE ballReset()
      ballReset();
    }
  }

  if (ballX > (pongWidth - ballSize / 2 - PADDLE_THICKNESS)) {
    if (ballY > paddle2Y && ballY < paddle2Y + PADDLE_HEIGHT) {
      ballSpeedX *= -1;
      var deltaY = ballY - (paddle2Y + PADDLE_HEIGHT / 2);
      ballSpeedY = deltaY * damp;
    } else {
      player1Score++; // must be BEFORE ballReset()
      ballReset();
    }
  }

  if (ballY > height) {
    ballY = height;
    ballSpeedY *= -1;
  }

  if (ballY < 0) {
    ballY = 0;
    ballSpeedY *= -1;
  }

  ballSpeedY = constrain (ballSpeedY, -5,5);
}

function computerMovement() {
  var paddle2YCenter = paddle2Y + PADDLE_HEIGHT / 2;
  if (paddle2YCenter < ballY - PADDLE_HEIGHT / 3) {
    paddle2Y += 6;
  } else if (paddle2YCenter > ballY + PADDLE_HEIGHT / 3) {
    paddle2Y -= 6;
  }
}

function ballReset() {
  if (player1Score >= WINNING_SCORE || player2Score >= WINNING_SCORE) {
    showingWinScreen = true;
  }
  ballSpeedX *= -1;
  ballX = pongWidth / 2;
  ballY = height / 2;
}

function drawEverything() {
  fill(255, 219, 0);
  rect(0, 0, PADDLE_THICKNESS,height);
  
  fill(255);
  noStroke();

  if (showingWinScreen) {
    textSize(20);
    if (player1Score >= WINNING_SCORE) {
      text("left player won!", 200, 200);
    } else if (player2Score >= WINNING_SCORE) {
      text("right player won!", pongWidth - 200, 200);
    }
    textSize(14);
    text("click to continue", pongWidth / 2, height - 200);
    return;
  }

  fill("#00B552");

  for (var i = 0; i < height; i += 40) {
    rect(pongWidth / 2 - 1, i, 2, 20);
  }

  rect(0, paddle1Y, PADDLE_THICKNESS, PADDLE_HEIGHT); // left paddle
  rect(pongWidth - PADDLE_THICKNESS, paddle2Y, PADDLE_THICKNESS, PADDLE_HEIGHT); // right paddle

  fill("#FAFBDF");
  ellipse(ballX, ballY, ballSize, ballSize); // ball //15 

  textSize(20);
  text(player1Score, 200, 100);
  text(player2Score, pongWidth - 200, 100);
}

function mouseReleased() {
  if (showingWinScreen) {
    player1Score = 0;
    player2Score = 0;
    showingWinScreen = false;
  }
}

function modelReady() {
  select('#loading').html('Model loaded!');
}

function predict() {
  regressor.predict(gotResults);
}

function createButtons() {
  slider = select('#slider');

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
  buttonPredict.mousePressed(function() {
    playGame = true;
    predict();
  });
}

// Show the results
function gotResults(err, result) {
  if (err) {
    console.error(err);
  }
  slider.value(result);
  updatePaddle(result);
  predict();
}

