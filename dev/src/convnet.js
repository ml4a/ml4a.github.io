function convnet(dataset_) 
{
  var self = this;
  var maxmin = cnnutil.maxmin;
  var f2t = cnnutil.f2t;
  var is=0, it=0;
  var dataset = dataset_;
  var layer_defs = [];
  var trainer;
  var net;
  var test_sample;
  var out_prob;
  var actual_label;
  var pred_label;
  var results = {confusion:[], tops:[], actuals:[], predictions:[]};
  
  layer_defs.push({type:'input', out_sx:dataset.get_dim(), out_sy:dataset.get_dim(), out_depth:dataset.get_channels()});

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
    return actual_label;
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

  this.setup = function() {
    net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:8, l2_decay:0.0001});
    //trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.001, momentum:0.0, batch_size:4, l2_decay:0.0002});
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

  this.train_next = function(n) {
    for (var t=is; t<is+n; t++) {
      var d = dataset.get_training_sample(t);
      trainer.train(d.vol, d.label);
    }
    console.log("finished training "+n+" samples");
    is += n;
  };

  this.test_next = function(n) {
    for (var t=it; t<it+n; t++) {
      

      test_sample = dataset.get_test_sample(t);
      out_prob = net.forward(test_sample.vol).w;
      var prob = Math.max.apply(Math, out_prob);

      a = test_sample.label;
      actual_label = a;
      p = out_prob.indexOf(prob);

      results.confusion[a][p] += 1;
      results.actuals[a] += 1;
      results.predictions[p] += 1;

//      console.log("IT IS "+max_prob +" :: "+ results.tops[actual_label][pred_label].prob);
      
      var max_tops = 10;

      var inserted = false;
      for (var i=0; i<results.tops[a][p].length; i++) {

        if (prob > results.tops[a][p][i].prob) {
          results.tops[a][p].splice(i, 0, {idx:t, prob:prob});
          inserted = true;
          break;
        }
      }
      if (!inserted && results.tops[a][p].length < max_tops) {
        results.tops[a][p].push({idx:t, prob:prob});
      }

      /*
      if (max_prob > results.tops[a][p].prob) {
        results.tops[a][p].idx = t;
        results.tops[a][p].prob = prob;
      }
      */

    }
    it += n;
    
  };

  this.get_weights_image = function() {
    var dim = dataset.dim;
    var nc = dataset.channels;
    var nw = net.layers[1]["filters"].length;
    var wimg = [];
    var min__ = +1e8;
    var max__ = -1e8;
    var min_=[], max_=[];
    for (var c=0; c<nc; c++) {
      max_.push(-1e8);
      min_.push(+1e8);
    }      
    for (var w=0; w<nw; w++) {
      var lw = net.layers[1]["filters"][w]["w"];
      for (var i=0; i<dim*dim; i++) {
        for (var c=0; c<nc; c++) {
          min_[c] = min(min_[c], lw[nc*i+c]);
          max_[c] = max(max_[c], lw[nc*i+c]);      
          min__ = min(min__, lw[nc*i+c]);
          max__ = max(max__, lw[nc*i+c]);      
        }
      }
      var img = createImage(dim, dim);
      img.loadPixels();  
      for (var i=0; i<dim*dim; i++) {
        for (var j=0; j<3; j++) { 
          var ic = nc == 1 ? 0 : j;
          var iw = nc * i + ic;
          //img.pixels[4*i+j] = map(lw[iw], min_[ic], max_[ic], 0, 255);
          img.pixels[4*i+j] = map(lw[iw], min__, max__, 0, 255);
        }
        img.pixels[4*i+3] = 255;
      }
      img.updatePixels();
      wimg.push(img);
    }
    return wimg;
  };

  this.get_test_sample_image = function() {
    var sample_img = test_sample == null ? null : dataset.get_image(test_sample);
    return sample_img;
  };

  this.get_sample_image = function(t) {
    var sample = dataset.get_test_sample(t);
    var sample_img = sample == null ? null : dataset.get_image(sample);
    return sample_img;
  };
};