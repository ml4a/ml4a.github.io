function disp_text(n, d) {
  var t = nfs(n, 0, d);
  return t[0] == ' ' ? t.slice(1) : t;
};

function draw_image_grid(img, x, y, w, h, sub) 
{	
  if (sub != null) {

  }
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
		//line(x * cell_size.w, 0, x * cell_size.w, dim * cell_size.h);
	}
	for (var y=0; y<dim; y++) {
		//line(0, y * cell_size.h, dim * cell_size.w, y * cell_size.h);
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

function draw_confusion(cn, x, y, w, h, iy, ix, is_tops) 
{
	var classes = cn.get_dataset().get_classes();
	var results = cn.get_results();
	var n = cn.get_dataset().get_classes().length;
	var cell_size = {w:w/n, h:h/n};
  var txtSize = 14;
  
	var min_precision_wrong = 1.0;
	var max_precision_wrong = 0.0;
	var min_precision_right = 1.0;
	var max_precision_right = 0.0;
  /*
	for (var a=0; a<n; a++) {
    for (var p=0; p<n; p++) {
      var raw = results.confusion[a][p] == 0 ? 1 : results.confusion[a][p];
      var precision = results.predictions[p] == 0 ? 0 : raw / results.predictions[p];
      var recall = results.actuals[a] == 0 ? 0 : raw / results.actuals[a];
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
  */
  min_precision_wrong = 0.0;
  max_precision_wrong = 1.0 - results.correct / results.total;

  push();
  textAlign(CENTER);
  translate(x, y);
	for (var a=0; a<n; a++) {
    for (var p=0; p<n; p++) {
      if (is_tops) {
        if (results.tops[a][p].length == 0) continue;
        var idx = results.tops[a][p][0].idx;
        var prob = results.tops[a][p][0].prob;
        var img = cn.get_test_sample_image(idx);
        push();
        translate(p * cell_size.w, a * cell_size.h);      
        noStroke();
        fill(a==p?0:255, a==p?255:0, 0);
        rect(0, 0, cell_size.w, cell_size.h);
        var imgSize = min(cell_size.w, cell_size.h) - 4;
        var imgMargin = {x:0.5 * (cell_size.w - imgSize), y:0.5 * (cell_size.h - imgSize)};
        image(img, imgMargin.x, imgMargin.y, imgSize, imgSize);
        pop();
      }
      else {
        var raw = results.confusion[a][p] == 0 ? 0 : results.confusion[a][p];
        var precision = results.predictions[p] == 0 ? 0 : raw / results.predictions[p];
        var recall = results.actuals[a] == 0 ? 0 : raw / results.actuals[a];
        push();
        translate(p * cell_size.w, a * cell_size.h);      
      	stroke(0, 10);
      	if (a == p) {
          var alpha = min_precision_right == max_precision_right ? 255 : map(precision, min_precision_right, max_precision_right, 90, 255);
  				fill(0, 255, 0, alpha);
      	}
        else {
          var alpha = min_precision_wrong == max_precision_wrong ? 0 : map(precision, min_precision_wrong, max_precision_wrong, 0, 255);
        	fill(255, 0, 0, alpha);
        }
        rect(0, 0, cell_size.w, cell_size.h);
        textSize(txtSize);
        fill(0);
        noStroke();        
        text(raw, 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
        pop();
      }
    }
  }
  textSize(txtSize);
  strokeWeight(2);
  stroke(0, 150);
  noFill();
  if (ix != -1 && iy != -1) {
    rect(ix * cell_size.w, iy * cell_size.h, cell_size.w, cell_size.h);
  }
  strokeWeight(1);
  for (var a=0; a<n; a++) {
    var pct = results.actuals[a] == 0 ? 0 : results.confusion[a][a] / results.actuals[a];
    push();    
    translate((n+0.5) * cell_size.w, a * cell_size.h);
    stroke(0, 150);
    fill(lerp(255, 0, pct), lerp(0, 255, pct), 0, 255);
    rect(0, 0, cell_size.w, cell_size.h);
    fill(0);
    noStroke();    
    text(results.actuals[a]==0?"":floor(100.0*pct)+"%", 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
    pop();
  }
  for (var p=0; p<n; p++) {
    var pct = results.predictions[p] == 0 ? 0 : results.confusion[p][p] / results.predictions[p];
    push();    
    translate(p * cell_size.w, (n+0.5) * cell_size.h);      
    stroke(0, 150);
    fill(lerp(255, 0, pct), lerp(0, 255, pct), 0, 255);
    rect(0, 0, cell_size.w, cell_size.h);
    fill(0);
    noStroke();    
    text(results.predictions[a]==0?"":floor(100.0*pct)+"%", 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
    pop();
  }
  
  var accuracy = results.total == 0 ? 0 : results.correct / results.total;
  push();
  translate((n+0.5) * cell_size.w, (n+0.5) * cell_size.h);      
  stroke(0, 150);
  fill(lerp(255, 0, accuracy), lerp(0, 255, accuracy), 0, 255);
  rect(-10, -10, cell_size.w+20, cell_size.h+20);
  fill(0);
  noStroke();    
  textStyle(BOLD);
  text("accuracy", 0.5 * cell_size.w, txtSize-6);
  textStyle(NORMAL);
  text(disp_text(100.0*accuracy,1)+"%", 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
  pop();

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
  textStyle(BOLD);
  push();
  textAlign(RIGHT);
  translate(-12, (n + 1.0) * cell_size.h);
  text("precision", 0, txtSize/2);
  pop();
	push();
  textAlign(LEFT);
  translate((n + 1.0) * cell_size.w, -8);
  rotate(-PI/8);    
  text("recall", 0, txtSize/2);
  pop();  
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
  var max_samples = 28;
  var sample_w = 64;
  push();
  translate(x, y);
  fill(0);
  noStroke();
  textStyle(BOLD);
  textSize(14);
  text(classes[actual]+(actual==predicted?" correctly classified as ":" misclassified as ")+classes[predicted], 5, 18);
  textAlign(CENTER);
  textStyle(ITALIC);
  textSize(11);
  text("hover mouse over confusion matrix to get top samples", w/2, -10);
  textSize(14);
  textStyle(NORMAL);
  for (var i=0; i<min(max_samples, n); i++) {
    var idx_pred_r = floor(i / 4);
    var idx_pred_c = i % 4;
  	var idx = results.tops[actual][predicted][i].idx;
  	var img = cn.get_test_sample_image(idx);
		var pct = floor(100.0*results.tops[actual][predicted][i].prob);
		push();
  	translate(4 + idx_pred_c * (sample_w+4), 24 + idx_pred_r * (sample_w+4+14));
  	image(img, 2, 2, sample_w, sample_w);
		text(pct+"%", 2 + 0.5*sample_w, sample_w+14);
  	pop();
  }
  pop();
  stroke(0);
  noFill();
  rect(x, y, w, h);
};

function draw_confusion_best(cn, max_top_samples, x, y, w, h) 
{
  fill(0);
  noStroke();
  textAlign(CENTER);
  textSize(16);
  textStyle(BOLD);
  text("Top mistakes", x + 0.5 * w, y-8);
  var classes = cn.get_dataset().get_classes();
  var results = cn.get_results();
  var w_rect = w;
  var w_height = 52;
  var best = [];
  for (var a=0; a<classes.length; a++) {
    for (var p=0; p<classes.length; p++) {
      if (a == p) continue;
      var inserted = false;
      for (var i=0; i<results.tops[a][p].length; i++) {
        var idx = results.tops[a][p][i].idx;
        var prob = results.tops[a][p][i].prob;
        for (var j=0; j<best.length; j++) {
          if (prob > best[j].prob) {
            best.splice(j, 0, {idx:idx, prob:prob, actual:a, predicted:p});
            best.splice(max_top_samples);
            inserted = true;
            break;
          }
        }
        if (!inserted && best.length < max_top_samples) {
          best.push({idx:idx, prob:prob, actual:a, predicted:p});
        }
      }
    }
  }
  textAlign(LEFT);
  textSize(12);
  textStyle(NORMAL);
  for (var j=0; j<best.length; j++) {
    var idx = best[j].idx;
    var prob = best[j].prob;
    var predicted = best[j].predicted;
    var actual = best[j].actual;
    var img = cn.get_test_sample_image(idx);
    push();
    translate(x, y + (w_height + 2) * j);
    image(img, 2, 2, 48, 48);
    noStroke();
    fill(0);
    text("actual: "+classes[actual], 54, 16);   
    text("predicted: "+classes[predicted], 54, 32);   
    text("confidence: "+floor(100*prob)+"%", 54, 48);   
    stroke(0, 100);
    noFill();
    rect(0, 0, w_rect, w_height);
    pop();
  }      
};

var idx_pred_r = 0;
var idx_pred_c = 0;
var idxt = -1;
function draw_last_predictions(cn, cols, x, y, w, h) 
{
  var idxt_ = cn.get_dataset().get_sample_index().test;
  if (idxt == idxt_) {
    return;
  }
  idxt = idxt_;
  var cell_width = w / cols;
  var cell_height = 36;
  var num_pred_cols = floor(w / cell_width);
  var classes = cn.get_dataset().get_classes();
  var simg = cn.get_test_sample_image();
  var a = cn.get_actual_label();
  var p = cn.get_predicted_label();
  push();
  translate(x + cell_width * idx_pred_c, y + cell_height * idx_pred_r);
  fill(a == p ? 0 : 255, a == p ? 255 : 0, 0);
  stroke(0);
  rect(1, 0, cell_width, cell_height);
  image(simg, 4, 2, 32, 32);
  noStroke();
  textSize(12);
  fill(0);
  if (a==p) {
    text(classes[p], 40, 24); 
  }
  else {
    fill(70);
    text(classes[p], 40, 16); 
    fill(0);
    text(classes[a], 40, 33);
    stroke(70);
    line(38, 12, 38 + textWidth(classes[p])+4, 12);    
  }
  pop();
  idx_pred_r += 1;
  if (idx_pred_r >= floor(h / cell_height)) {
    idx_pred_r = 0;
    idx_pred_c = (idx_pred_c + 1) % num_pred_cols;
  }
}


///----------------

/*
function draw_activations(A, scale) {
  var act_images = [];

  var maxmin = cnnutil.maxmin;
  var f2t = cnnutil.f2t;

  var s = scale || 2; // scale
  var draw_grads = false;
  if(typeof(grads) !== 'undefined') draw_grads = grads;
  
  // get max and min activation to scale the maps automatically
  var w = draw_grads ? A.dw : A.w;
  var mm = maxmin(w);

  // create the canvas elements, draw and add to DOM
  for(var d=0;d<A.depth;d++) {
    var W = A.sx * s;
    var H = A.sy * s;

    var img = createImage(A.sx * s, A.sy * s);
    img.loadPixels();
    
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
            img.pixels[pp  ] = dval;  // rgb
            img.pixels[pp+1] = dval;  // rgb
            img.pixels[pp+2] = dval;  // rgb
            img.pixels[pp+3] = 255;   // alpha
          }
        }
      }
    }    
    img.updatePixels();
    act_images.push(img);
  }
  return act_images;
}
*/

