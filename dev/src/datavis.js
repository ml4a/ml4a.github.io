function draw_image_grid(img, x, y, w, h) 
{	
	var dim_x = img.width;
	var dim_y = img.height;
	var cell_size = {w:w/dim_x, h:h/dim_y};
	var text_ratio = 0.55;
	push();
	translate(x, y);
	textAlign(CENTER);
	textSize(cell_size.h * text_ratio);
	noFill();
	stroke(0, 50);
	strokeWeight(1);
	for (var x=0; x<dim; x++) {
		line(x * cell_size.w, 0, x * cell_size.w, dim * cell_size.h);
	}
	for (var y=0; y<dim; y++) {
		line(0, y * cell_size.h, dim * cell_size.w, y * cell_size.h);
	}
	stroke(0, 80);
	rectMode(CORNER);
	rect(0, 0, dim * cell_size.w, dim * cell_size.h);
	for (var y=0; y<dim; y++) {
		for (var x=0; x<dim; x++) {
			var p_idx = 4 * (x + dim * y);
			var c = img.get(x, y);
			var value = red(c);
			push();
			translate((x+0.5) * cell_size.w, (y+0.5) * cell_size.h);
			fill(0, map(value, 0, 255, 205, 255));
			noStroke();
			text(value, 0, 0.5 * text_ratio * cell_size.h);
			pop();
		}
	}  
	pop();
};

function draw_confusion(cn, x, y, w, h, iy, ix) 
{
	var classes = cn.get_dataset().get_classes();
	var results = cn.get_results();
	var n = cn.get_dataset().get_classes().length;
	var cell_size = {w:w/n, h:h/n};
	var txtSize = 12;
	var min_precision_wrong = 1.0;
	var max_precision_wrong = 0.0;
	var min_precision_right = 1.0;
	var max_precision_right = 0.0;
	for (var a=0; a<n; a++) {
    for (var p=0; p<n; p++) {
      var raw = results.confusion[a][p];
      var precision = raw / results.predictions[p];
      var recall = raw / results.actuals[a];
    	if (a==p) {
				min_precision_right = min(precision, min_precision_right);
	    	max_precision_right = max(precision, max_precision_right);
    	}
    	else {
	      min_precision_wrong = min(precision, min_precision_wrong);
	    	max_precision_wrong = max(precision, max_precision_wrong);
	    }
		}	
	}
  push();
  textAlign(CENTER);
  translate(x, y);
	for (var a=0; a<n; a++) {
    for (var p=0; p<n; p++) {
      var raw = results.confusion[a][p];
      var precision = raw / results.predictions[p];
      var recall = raw / results.actuals[a];
      push();
      translate(p * cell_size.w, a * cell_size.h);      
    	noStroke();
    	if (a == p) {
				fill(0, 255, 0, map(precision, min_precision_right, max_precision_right, 90, 255));
    	}
      else {
      	fill(255, 0, 0, map(precision, min_precision_wrong, max_precision_wrong, 0, 200));
      }
      rect(0, 0, cell_size.w, cell_size.h);
      fill(0);
      noStroke();
      text(raw, 24, 16);
      pop();
    }
  }
  if (ix != -1 && iy != -1) {
  	stroke(0, 150);
    strokeWeight(2);
    noFill();
    rect(ix * cell_size.w, iy * cell_size.h, cell_size.w, cell_size.h);
  }
  noStroke(0);
  fill(0);
  textSize(txtSize);
  textAlign(RIGHT);
  for (var a=0; a<n; a++) {
  	push();
    translate(-12, (a + 0.5) * cell_size.h);
    textStyle(a == iy ? BOLD : NORMAL);
    text("actual "+classes[a], 0, txtSize/2);
    pop();
	}
	textAlign(LEFT);
	for (var p=0; p<n; p++) {
  	push();
    translate((p + 0.5) * cell_size.w, -4);
    rotate(-PI/8);
    textStyle(p == ix ? BOLD : NORMAL);
    text("predicted "+classes[p], 0, 0);
    pop();
  }
  pop();

  stroke(0);
  noFill();
  rect(x, y, w, h);
};

function draw_confusion_samples(cn, x, y, w, h, actual, predicted) 
{
	var classes = cn.get_dataset().get_classes();
	var results = cn.get_results();
  var n = results.tops[actual][predicted].length;
  var max_samples = 8;
  var sample_w = (w - 8) / max_samples;
  push();
  translate(x, y);
  fill(0);
  noStroke();
  textStyle(BOLD);
  text(classes[actual]+(actual==predicted?" correctly classified as ":" misclassified as ")+classes[predicted], 5, 18);
  textAlign(CENTER);
  textStyle(NORMAL);
  for (var i=0; i<min(max_samples, n); i++) {
  	var idx = results.tops[actual][predicted][i].idx;
  	var img = cn.get_sample_image(idx);
		var pct = floor(100.0*results.tops[actual][predicted][i].prob);
		push();
  	translate(4 + i * sample_w, 22);
  	image(img, 0, 0, sample_w - 2, sample_w - 2);
		text(pct+"%", (sample_w-2) / 2, sample_w + 13);
  	pop();
  }
  pop();
  stroke(0);
  noFill();
  rect(x, y, w, h);
};

function draw_confusion_sample_grid(cn, x, y, w, h) 
{
	var classes = cn.get_dataset().get_classes();
	var results = cn.get_results();
	var n = cn.get_dataset().get_classes().length;
	var cell_size = {w:w/n, h:h/n};
	var txtSize = 14;
  push();
  textAlign(CENTER);
  translate(x, y);
	for (var a=0; a<n; a++) {
    for (var p=0; p<n; p++) {
    	if (results.tops[a][p].length == 0) continue;
    	var idx = results.tops[a][p][0].idx;
    	var prob = results.tops[a][p][0].prob;
  		var img = cn.get_sample_image(idx);
      push();
      translate(p * cell_size.w, a * cell_size.h);      
    	noStroke();
    	fill(a==p?0:255, a==p?255:0, 0);
      rect(0, 0, cell_size.w, cell_size.h);
      image(img, 2, 2, cell_size.w-4, cell_size.h-4);
      pop();
    }
  }
  noStroke(0);
  fill(0);
  textSize(txtSize);
  textAlign(RIGHT);
  for (var a=0; a<n; a++) {
  	push();
    translate(-12, (a + 0.5) * cell_size.h);
    text("actual "+classes[a], 0, txtSize/2);
    pop();
	}
	textAlign(LEFT);
	for (var p=0; p<n; p++) {
  	push();
    translate((p + 0.5) * cell_size.w, -4);
    rotate(-PI/8);
    text("predicted "+classes[p], 0, 0);
    pop();
  }
  pop();
};





///----------------


function draw_activations(A, scale) {
  
  //var s = scale || 2; // scale
  var s = 1;
  var draw_grads = false;
  if(typeof(grads) !== 'undefined') draw_grads = grads;
  
  // get max and min activation to scale the maps automatically
  var w = draw_grads ? A.dw : A.w;
  var mm = maxmin(w);

  
  // create the canvas elements, draw and add to DOM
  for(var d=0;d<A.depth;d++) {
    var W = A.sx * s;
    var H = A.sy * s;

    var img_ = createImage(A.sx, A.sy);
    img_.loadPixels();
    
    for(var x=0;x<A.sx;x++) {
      for(var y=0;y<A.sy;y++) {
        if(draw_grads) {
          var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
        } else {
          var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
        }
        for(var dx=0;dx<s;dx++) {
          for(var dy=0;dy<s;dy++) {
            var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
            img_.pixels[pp  ] = dval;  // rgb
            img_.pixels[pp+1] = dval;  // rgb
            img_.pixels[pp+2] = dval;  // rgb
            img_.pixels[pp+3] = 255;   // alpha
          }
        }
      }
    }
    
    img_.updatePixels();
    
    var yy = floor(d / 14) * 82;// (A.sy + 2);
    var xx = (d % 14) * 82;//(A.sx + 5);
    
    push();
    translate(xx, yy);
    noSmooth();
    image(img_, 0, 0, 80, 80);
    pop();
  }
}


