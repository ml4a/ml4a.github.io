function neuron() 
{
	// style
  var rectangle = {x: 0, y: 0, w: 0, h: 0};
  var h1, h2, h3;
  var radius;
  this.value_visible = true;
  this.bias_visible = true;
  this.bold = true;

  // state
  this.bias = 0.0;
  this.value = 0.0;

	this.set_rectangle = function(x, y, w, h) {
  	rectangle = {x: x, y: y, w: w, h: h};
  };

  this.set_radius = function(radius_) {
  	radius = radius_;
  	h1 = radius / 1.5;
  	h2 = h1 / 2.0;
    h3 = radius * 1.4;
  };

  this.get_rectangle = function() {
  	return rectangle;
  };

  this.get_center = function() {
  	return {x: rectangle.x + rectangle.w/2, y: rectangle.y + rectangle.h/2};
  };

  this.get_radius = function() {
  	return radius;
  };
  
  this.draw = function(txt, draw_arrow) {
    var center = this.get_center();
		push();
		translate(center.x, center.y);
  	if (this.bold) {
  		stroke(0);
    	strokeWeight(3);
  	}
  	else {
  		stroke(0, 155);
  		strokeWeight(1.5);
  	}
  	fill(255);
  	ellipse(0, 0, 2 * radius, 2 * radius);
		noStroke();
		textAlign(CENTER);
    if (this.value_visible) {
    	fill(0, this.bold ? 255 : 170);
			textSize(h1);
			text(disp_text(this.value, 2), 0, h1/4);
		}
		if (this.bias_visible && this.bias != null) {
	    fill(0, this.bold ? 180 : 100);
	    textSize(h2);
	    text("b = "+disp_text(this.bias, 2), 0, -0.8 * radius + h2);
    }
    if (txt != "") {
      push();
    	fill(0);
    	noStroke();
      textSize(h3);
    	textAlign(RIGHT);
      if (draw_arrow) {
        text(txt, -radius - 17, radius - h3/2.0 + 1);
        stroke(0, 150);
        strokeWeight(1);
        line(-radius - 15, 0, -radius - 4, 0);
        line(-radius - 4, 0, -radius - 7, -2);
        line(-radius - 4, 0, -radius - 7, +2);
      }
      else {
        text(txt, -radius - 7, radius - h3/2.0 + 1);
      }
    	pop();
    }
	  pop();
  };

  this.set_radius(30);
};


function layer(n, maxn_) 
{
	this.neurons = [];
	this.visible = [];
	this.weights = [];
	this.weights_visible = [];
	var rectangle = {x: 0, y:0, w: 0, h: 0};
  var maxn = maxn_;
	var next_layer;

	this.set_rectangle = function(x, y, w, h) {
    rectangle = {x: x, y: y, w: w, h: h};
    this.set_positions();
  };

	this.get_rectangle = function() {
  	return rectangle;
  };

  this.get_center = function() {
  	return {x: rectangle.x + rectangle.w / 2, y: rectangle.y + rectangle.h / 2};
  };

  this.set_maxn = function(maxn_) {
  	maxn = maxn_;
  	this.visible = [];
  	for (var i = 0; i < this.neurons.length; i++) {
			if (i < maxn) {
      	this.visible.push(i);
      }
    }
    if (n > maxn) {
			this.visible[maxn-1] = n-1;
    }
    this.set_positions();
  }

  /////////// SET CUSTOM VISIBLE
  //
  //
  //
  this.set_vis = function(vis) {
    maxn = 10; //vis.length;
    this.visible = [];

    for (var i = 11; i < 18; i++) {
      for (var j = 11; j < 18; j++) {
        this.visible.push(j + 28*i);;
      }
    }

    this.set_positions();
  };



  this.set_neuron_style = function(radius, bold_) {
  	for (var j=0; j<this.neurons.length; j++) {
  		this.neurons[j].set_radius(radius);
  		this.neurons[j].bold = bold_;
		}
  };

	this.add_neurons = function(n) {
		for (var i = 0; i < n; i++) {
			var newneuron = new neuron();
      this.neurons.push(newneuron);
    }
    this.set_maxn(maxn);
  };
  
	this.forward_connect = function(next_layer_) {
    next_layer = next_layer_;
    for (var i = 0; i < this.neurons.length; i++) {
    	var neuron_weights = [];
    	var neuron_weights_visible = [];
    	for (var j = 0; j < next_layer.neurons.length; j++) {
      	neuron_weights.push(random(1));
      	neuron_weights_visible.push(true);
    	}
    	this.weights.push(neuron_weights);
    	this.weights_visible.push(neuron_weights_visible);
    }
	};

	this.set_positions = function() {
    var r = rectangle;
    var v = this.visible;
    for (var i=0; i<v.length; i++) {
			var i_ = v[i];
    	var x = r.x;
    	var y = map(i, 0, v.length, r.y, r.y + r.h);
    	var w = r.w;
    	var h = r.h / v.length;    	
    	if (v.length < this.neurons.length) {
      	var last = (i == v.length-1 ? 1 : 0);
      	y = map(i + last, 0, 1 + v.length, r.y, r.y + r.h);
      	h = r.h / (1 + v.length);
    	}
      this.neurons[i_].set_rectangle(x, y, w, h);
    }
	};

	this.add_neurons(n);
};


function nnvis(net_) 
{
	var layers = [];
	var layer_draw_types = [];
	var out_prob = [];
	var actual_label = 0;
	var rectangle = {x: 0, y: 0, w: 0, h: 0};
	var draw_weight_arrow = true;
  var draw_input_arrow = true;
  var input_label = "";
  var label_max_width;
  
	this.layer_draw_type = { NORMAL : 1, PIXEL: 2 };

	this.set_layer_style = function(layer_num, draw_type, maxn, radius, bold_) {
		layers[layer_num].set_maxn(maxn);
		if (draw_type == this.layer_draw_type.PIXEL) {
			layers[layer_num].set_neuron_style(0, false);
		}
		else {
			layers[layer_num].set_neuron_style(radius, bold_);
		}
		layer_draw_types[layer_num] = draw_type;
	};

  this.set_input_label = function(input_label_) {
    input_label = input_label_;
  }

  this.set_rectangle = function(x, y, w, h) {
    rectangle = {x: x, y: y, w: w, h: h};
    this.set_positions();
  };

  this.get_rectangle = function() {
  	return rectangle;
  };

  this.get_center = function() {
  	return {x: rectangle.x + rectangle.w / 2, y: rectangle.y + rectangle.h / 2};
  };

  this.get_layer = function(l) {
  	return layers[l];
  };

  this.num_layers = function() {
    return layers.length;
  };

	this.get_draw_weight_arrow = function() {
  	return draw_weight_arrow;
  };
  
  this.get_draw_input_arrow = function() {
    return draw_input_arrow;
  };

  this.set_neuron_style = function(radius, bold_) {
		for (var i=0; i<layers.length; i++) {
			layers[i].set_neuron_style(radius, bold_);
		}
  };

	this.set_draw_weight_arrow = function(draw_weight_arrow_) {
  	draw_weight_arrow = draw_weight_arrow_;
  };

  this.set_draw_input_arrow = function(draw_input_arrow_) {
    draw_input_arrow = draw_input_arrow_;
  };

  this.set_positions = function() {
    for (var i=0; i<layers.length; i++) {
      var x = map(i, 0, layers.length, rectangle.x, rectangle.x + rectangle.w);
      var y = rectangle.y;
      var w = rectangle.w / layers.length;
      var h = rectangle.h;
      layers[i].set_rectangle(x, y, w, h);
    }
  };

  this.set_neuron_info_visible = function(value_visible_, bias_visible_) {
    for (var i=0; i<this.num_layers(); i++) {  
      var n = this.get_layer(i).neurons;
      for (var j=0; j<n.length; j++) {
        n[j].value_visible = value_visible_;
        n[j].bias_visible = bias_visible_;               
      }
    }
  };

  this.highlight_weights = function(idx_l, idx_n, v) {
    for (var i=0; i<this.num_layers()-1; i++) {
      var l1 = this.get_layer(i);
      var l2 = this.get_layer(i+1);
      for (var j=0; j<l1.visible.length; j++) {
        var j_ = l1.visible[j];
        for (var k = 0; k < l2.visible.length; k++) {
          var k_ = l2.visible[k];
          var s = (i+1==idx_l) && (k_ == idx_n);
          l1.weights_visible[j_][k_] = v ? s : !s;
        }
      }
    }
  };

  this.add_layer = function(n, maxvisible) {
    var newlayer = new layer(n, maxvisible);
    layers.push(newlayer);
    layer_draw_types.push(this.layer_draw_type.NORMAL);
    this.set_positions();
    if (layers.length > 1) {
    	var prevlayer = layers[layers.length - 2];
    	prevlayer.forward_connect(newlayer);
    }
    else {
    	for (var i=0; i<newlayer.neurons.length; i++) {
    		newlayer.neurons[i].bias = null;
    	}
    }
  };

  this.setup_from_net = function() {
  	var layer_defs = this.net.get_layer_defs();
  	for (var i=0; i<layer_defs.length; i++) {
	 		var type = layer_defs[i].type;
	 		if (type == 'input') {
	 			var n = layer_defs[i].out_sx * layer_defs[i].out_sy;
	 			this.add_layer(n, 10);
	 		}
	 		else if (type == 'fc') {
	 			var n = layer_defs[i].num_neurons;
	 			this.add_layer(n, 10);
	 		}
	 		else if (type == 'softmax') {
	 			var n = layer_defs[i].num_classes;
	 			this.add_layer(n, 10);
  		}
	  }
  };

	this.forward_propagate = function() {
  	for (var i=0; i<layers.length-1; i++) {
  		var l1 = layers[i];
  		var l2 = layers[i+1];
  		for (var j = 0; j < l2.neurons.length; j++) {
	  		var total = 0;
	  		for (var k=0; k<l1.neurons.length; k++) {
	    		total += (l1.neurons[k].value * l1.weights[k][j]);
	  		}
	  		total += l2.neurons[j].bias;
	  		l2.neurons[j].input = total;
	  		l2.neurons[j].value = sigmoid(total);
	  	}
  	}
	};

	this.update = function() {
		var idxl = layers.length-1;
		var idxn = this.net.get_net().layers.length-1;
		var acts = this.net.get_out_acts(idxn);
		var prob = this.net.get_prob();
		actual_label = this.net.get_actual_label();
		for (var j=0; j<layers[idxl].neurons.length; j++) {
			layers[idxl].neurons[j].value = acts[j];
		}
    for (var i=0; i<prob.length; i++) {
       out_prob[i] = prob[i];
    }
	};

	this.draw_network = function() {  	
		for (var i=0; i<layers.length; i++) {
			var l = layers[i];
 			var v = l.visible;
 			if (layer_draw_types[i] == this.layer_draw_type.PIXEL) {
 				noStroke();
 				var img = this.net.get_test_sample_image();
				if (img != null) {
					img.loadPixels();
				}
 				for (var j=0; j<l.neurons.length; j++) {
 					var x1 = l.neurons[0].get_center().x;
 					var y1 = map(j, 0, l.neurons.length, rectangle.y, rectangle.y+rectangle.h);
 					if (img != null) {
 						fill(img.pixels[4*j], img.pixels[4*j+1], img.pixels[4*j+2]);
					}
					else {
						fill(255);
					}
 					//noStroke();
      		rect(x1-4, y1, 2, 1);    
 				}
 			}
 			else if (layer_draw_types[i] == this.layer_draw_type.NORMAL) {
	    	for (var j=0; j<v.length; j++) {
          var label = (i==0 && input_label != "") ? input_label+" "+(1+v[j]) : "";
	    		l.neurons[v[j]].draw(label, draw_input_arrow);
	    	}
	    }
    	if (v.length < l.neurons.length) {
				var r = l.get_rectangle();
				var x = r.x + r.w/2;
				var y = map(v.length-0.5, 0, v.length+1, r.y, r.y + r.h);
				var w = r.w;
	      var h = r.h / v.length;
				push();
				textSize(75);
				fill(0);
				noStroke();
				ellipse(x, y-h/4, h/6, h/6);
				ellipse(x, y, 	  h/6, h/6);
				ellipse(x, y+h/4, h/6, h/6);
				pop();
			}
  	}
  	for (var i=0; i<layers.length-1; i++) {
  		var l1 = layers[i];
  		var l2 = layers[i+1];
			var other_visible = l2.visible;
			var dj = layer_draw_types[i] == this.layer_draw_type.PIXEL ? 10 : 1;
			for (var j=0; j<l1.visible.length; j+=dj) {
				var j_ = l1.visible[j];
				var n1 = l1.neurons[j_];
				for (var k = 0; k < other_visible.length; k++) {
					var k_ = other_visible[k];
					var n2 = l2.neurons[k_];
					var n1c = n1.get_center();
					var n2c = n2.get_center();
					var ang = atan2(n2c.y - n1c.y, n2c.x - n1c.x);
					var distance = dist(n1c.x, n1c.y, n2c.x, n2c.y)
					var rad1 = n1.get_radius();
					var rad2 = n2.get_radius();
					var arrowlen = draw_weight_arrow ? constrain(map(l1.visible.length, 0, 120, 10, 0), 0, 10) : 0;
					var alpha = constrain(map(l1.visible.length, 0, 120, 80, 10), 40, 80);
					push();
					translate(n1c.x, n1c.y);
					rotate(ang);
					noFill();
					if (l1.weights_visible[j_][k_]) {
						stroke(0);
						strokeWeight(2);
					}
					else {
						stroke(0, alpha);
						strokeWeight(1);
					}


          ////////////////
          /*
          var w = net.get_net().layers[1].filters[k].w;
          var mm = net.get_maxmin(w);
          var sss = ( net.get_net().layers[1].filters[k].w[j] - mm.minv ) / mm.dv;
          //strokeWeight(map(sss, 0, 1, 0, 15));
          strokeWeight(5*sss);
          stroke(0, 175*sss);
          //stroke(100*sss, 100, 100, 200);
          */
          
				  line(n1.get_radius(), 0, distance - n2.get_radius(), 0);
					line(distance - n2.get_radius(), 0, distance - n2.get_radius() - arrowlen, -arrowlen/2);
					line(distance - n2.get_radius(), 0, distance - n2.get_radius() - arrowlen, +arrowlen/2);
					if (l1.weights_visible[j_][k_]) {
						noStroke();
						fill(0);
						text("w = "+disp_text(l1.weights[j_][k_], 2), n1.get_radius() + 0.03 * (distance - n1.get_radius()), -2);
					}
					pop();
				}
			}      		
  	}
	};

  this.draw_result = function() {
    var outv = layers[layers.length-1].visible;
    var outn = layers[layers.length-1].neurons;
    var classes = this.net.get_dataset().get_classes();
    for (var i=0; i<outv.length; i++) {
      var center = outn[outv[i]].get_center();
      var x = center.x + label_max_width + outn[outv[i]].get_radius();
      var y = center.y;
      var w = map(out_prob[outv[i]], 0, 1, 0, 100);
      var h = 20;
      push();
      noStroke();
      fill(0);
      textSize(12);
      textAlign(RIGHT);
      text(classes[outv[i]], x-4, y + 5);
      if (outv[i] == actual_label) {
        fill(0, 255, 0);
      }
      else {
        fill(255, 0, 0);  
      }
      rect(x, y-h/2, w, h);
      pop();
    }   
  };

	this.initialize = function() {
		for (var i=0; i<layers.length-1; i++) {
			var l1 = layers[i];
			for (var j=0; j<l1.neurons.length; j++) {
				l1.neurons[j].value_visible = false;
				l1.neurons[j].bias_visible = false;
			}
			var l2 = layers[i+1];
			for (var j=0; j<l1.neurons.length; j++) {
				for (var k=0; k<l2.neurons.length; k++) {
					l1.weights_visible[j][k] = false;
				}
			}
			if (i == layers.length-2) {
				for (var j=0; j<l2.neurons.length; j++) {
					l2.neurons[j].bias_visible = false;
				}
			}		
		}
		this.randomize_network();
	};

  this.randomize_network = function() {
		for (var i=0; i<layers.length; i++) {
			var l1 = layers[i];
			for (var j=0; j<l1.neurons.length; j++) {
				l1.neurons[j].bias = (i==0 ? null : ((random(1) < 0.5 ? 1 : -1) * random(0.05, 0.5)));
  			l1.neurons[j].value = (random(1) < 0.5 ? 1 : -1) * random(0.05, 0.5);
  		}
			if (i < layers.length-1) {
				var l2 = layers[i+1];
				for (var j=0; j<l1.neurons.length; j++) {
					for (var k = 0; k < l2.neurons.length; k++) {
						l1.weights[j][k] = random(1);
					}
				}
			}
		}
	};

  this.get_neuron_compute_string = function(l, n) {
    var s = "z = ";
    for (var i=0; i<layers[l-1].neurons.length; i++) {
      var w = disp_text(layers[l-1].weights[i][n], 2);
      var v = disp_text(layers[l-1].neurons[i].value, 2);
      s += (w+" * "+v)+" + ";
    }
    s += disp_text(layers[l].neurons[n].bias, 2);
    s += "\n";
    s += "value = 1 / (1 + exp(-z)) = "+disp_text(layers[l].neurons[n].value,2);
    //s = s.slice(s.length-1);
    //"z = "+disp_text(vis.get_layer(0).neurons[0].value, 2)+" * "+disp_text(vis.get_layer(0).weights[0][1], 2)
    return s;
  };

	this.process_mouse = function(mx, my) {
  	for (var i=0; i<layers.length; i++) {
    	for (var j=0; j<layers[i].neurons.length; j++) {
      	var d = dist(mx, my, layers[i].neurons[j].position.x, layers[i].neurons[j].position.y);
      	if (d < neuronRadius) {
      		// do something with mouse (mx, my), neuron (i, j)
       		return;
      	}
    	}
  	}
	};

  this.draw_sample = function() {
    var img = this.net.get_test_sample_image();
    image(img, 20, 32, 100, 100);
  };


	/* initial setup */
	if (net_ != null) {
		this.net = net_;
		this.setup_from_net(this.net);
    label_max_width = 64;
		this.get_net = function() {return this.net;}
  };
  this.initialize();
};


function sigmoid(z) {
  return 1.0 / (1.0 + exp(-z));
};


function disp_text(n, d) {
	var t = nfs(n, 0, d);
	return t[0] == ' ' ? t.slice(1) : t;
};
