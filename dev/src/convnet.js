function convnet(dataset_) 
{
  var self = this;
  var maxmin = cnnutil.maxmin;
  var f2t = cnnutil.f2t;
  var dataset = dataset_;

  var layer_defs = [];
  var trainer;
  var net;

  var results;
  var max_tops = 32;

  var test_idx;
  var out_prob;
  var a, p;
  
  this.add_layer = function(layer_def) {
    layer_defs.push(layer_def);
  };

  this.get_layer_defs = function() {
    return layer_defs;
  };
  
  this.get_prob = function() {
    return out_prob;
  };

  this.get_actual_label = function() {
    return a;
  };

  this.get_predicted_label = function() {
    return p;
  };

  this.get_in_acts = function(l) {
    return net.layers[l].in_act.w;
  };
  
  this.get_out_acts = function(l) {
    return net.layers[l].out_act.w;
  };

  this.get_net = function() {
    return net;
  };

  this.get_dataset = function() {
    return dataset;
  };

  this.get_results = function() {
    return results;
  };

  this.load_from_json = function(json) {
    net = new convnetjs.Net();
    net.fromJSON(json);
  };

  this.setup_trainer = function(trainer_params) {
    net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    trainer = new convnetjs.SGDTrainer(net, trainer_params);
  };

  var setup_results = function() {
    results = {confusion:[], tops:[], actuals:[], predictions:[], correct:0, total:0};
    var n = dataset.get_classes().length;
    for (var i=0; i<n; i++) {
      var crow = [];
      var trow = [];
      for (var j=0; j<n; j++) {
        crow.push(0);
        trow.push([]);
      }
      results.confusion.push(crow);
      results.tops.push(trow);
      results.predictions.push(0);
      results.actuals.push(0);
    }
  };

  this.train_next = function() {
    var sample = dataset.get_next_training_sample();
    if (sample == null) { 
      return false;
    }
    else {
      trainer.train(sample.vol, sample.label);
      return true;
    }
  };

  this.train_all = function(callback) {    
    var is_training = true;
    while (is_training) {
      is_training = this.train_next();
      if (!is_training) {
        if (!dataset.is_fully_loaded()) {
          dataset.request_next_batch(function() {
            self.train_all(callback);
          });
        }
        else if (callback != null) {
          callback();
        }
      }
    }
  };

  this.test_all = function(callback) {  
    var is_testing = true;
    while (is_testing) {
      is_testing = this.test_next();
      if (!is_testing) {
        if (!dataset.finished_testing()) {
          dataset.request_next_batch(function() {
            self.test_all(callback);
          });
        }
        else if (callback != null) {
          callback();
        }
      }
    }
  };

  this.test_next = function() {
    var sample = dataset.get_next_test_sample();
    if (sample == null) {
      if (!dataset.finished_testing()) {
        dataset.request_next_batch(function (){
          return self.test_next();
        });
      }
      return false;
    }    
    test_idx = sample.idx;

    // problems in firefox?
    /*
    out_prob = net.forward(sample.vol).w;
    var prob = Math.max.apply(Math, out_prob);
    a = sample.label;
    p = out_prob.indexOf(prob);    
    */

    // this is poor form, but works...
    out_prob = net.forward(sample.vol).w;
    var prob = 0;
    p = 0;
    for (var i=0; i<out_prob.length; i++) {
      if (out_prob[i] > prob) {
        p = i;
        prob = out_prob[i];
      }
    }
    a = sample.label;

    results.confusion[a][p] += 1;
    results.actuals[a] += 1;
    results.predictions[p] += 1;
    results.total += 1;
    results.correct += (a==p?1:0);    
    var inserted = false;
    for (var i=0; i<results.tops[a][p].length; i++) {
      if (prob > results.tops[a][p][i].prob) {
        results.tops[a][p].splice(i, 0, {idx:test_idx, prob:prob});
        inserted = true;
        break;
      }
    }
    if (!inserted && results.tops[a][p].length < max_tops) {
      results.tops[a][p].push({idx:test_idx, prob:prob});
    }
    return true;
  };

  this.start_over = function() {
    setup_results();
    dataset.set_test_index(0);
  };

  this.save_dataset = function() {
    var data = JSON.stringify(net.toJSON());
    var url = 'data:text/json;charset=utf8,' + encodeURIComponent(data);
    window.open(url, '_blank');
    window.focus();
  };

  this.array_to_image = function(A, idx, scale, draw_grads) {
    //var nc = dataset.channels; // check this
    var s = scale || 1; // scale
    var dg = draw_grads || false; // draw grads
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);
    var W = A.sx * s;
    var H = A.sy * s;
    var img = createImage(W, H);
    img.loadPixels();    
    for(var x=0;x<A.sx;x++) {
      for(var y=0;y<A.sy;y++) {
        var val = Math.floor(((dg?A.get_grad(x,y,idx):A.get(x,y,idx))-mm.minv)/mm.dv*255);
        for(var dx=0;dx<s;dx++) {
          for(var dy=0;dy<s;dy++) {
            var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
            img.pixels[pp  ] = val;
            img.pixels[pp+1] = val;
            img.pixels[pp+2] = val;
            img.pixels[pp+3] = 255;
          }
        }
      }
    }    
    img.updatePixels();
    return img;
  };

  this.get_activations_image = function(layer, idx, scale, draw_grads) {
    var A = net.layers[layer].out_act;
    return this.array_to_image(A, idx, scale, draw_grads);
  };

  this.get_weights_image = function(idx, scale, draw_grads) {
    var A = net.layers[1].filters[idx];
    return this.array_to_image(A, idx, scale, draw_grads);
  };

  this.get_num_activations = function(layer) {
    return net.layers[1].out_act.depth;
  };
  
  this.get_num_weights = function(layer) {
    return net.layers[1].filters.length;
  };
  
  this.get_training_sample_image = function(t) {
    var sample = dataset.get_training_sample(t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };

  this.get_test_sample_image = function(t) {
    var sample = dataset.get_test_sample(t == null ? test_idx : t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };

  // initialize
  layer_defs.push({type:'input', out_sx:dataset.get_dim(), out_sy:dataset.get_dim(), out_depth:dataset.get_channels()});
  setup_results();
};