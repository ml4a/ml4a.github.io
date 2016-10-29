// TODO
//  - augmentation
//  - 

function convnet(dataset_) 
{
    this.get_net = function() {return net;}
    this.get_summary = function(){return summary;}
    this.get_num_activations = function(layer) {return net.layers[layer].out_act.depth;};
    this.get_num_filters = function(layer) {return net.layers[layer].filters.length;};

    this.add_layer = function(layer_def) {
        layer_defs.push(layer_def);
    };

    this.setup_trainer = function(trainer_params) {
        net = new convnetjs.Net();
        net.makeLayers(layer_defs);
        trainer = new convnetjs.SGDTrainer(net, trainer_params);
    };

    this.save_dataset = function() {
        var data = JSON.stringify(net.toJSON());
        var url = 'data:text/json;charset=utf8,' + encodeURIComponent(data);
        window.open(url, '_blank');
        window.focus();
    };

    this.save_summary = function() {
        var data = JSON.stringify(summary);
        var url = 'data:text/json;charset=utf8,' + encodeURIComponent(data);
        window.open(url, '_blank');
        window.focus();
    };

    this.load_summary = function(summary_file, callback) {
        $.getJSON(summary_file, function(json){
            summary = json;
            callback();
        });
    };

    this.load_from_json = function(snapshot_file, callback) {
        $.getJSON(snapshot_file, function(json){
            net = new convnetjs.Net();
            net.fromJSON(json);
            callback();
        });
    };

    function check_if_ready(callback) {
        if (!dataset.labelsLoaded) {
            dataset.load_labels(function(){
                callback(0);
            });
        } else {
            callback(0);
        }  
    };

    function test_sample(sample) {
        var out_prob = net.forward(sample.x).w;
        var top_prob = Math.max.apply(Math, out_prob);
        var p = out_prob.indexOf(top_prob); 
        var a = sample.y;   
        summary.confusion[a][p] += 1;
        summary.actuals[a] += 1;
        summary.predictions[p] += 1;
        summary.total += 1;
        summary.correct += (a==p?1:0);    
        var inserted = false;
        for (var i=0; i<summary.tops[a][p].length; i++) {
            if (top_prob > summary.tops[a][p][i].prob) {
                summary.tops[a][p].splice(i, 0, {idx:sample.idx, prob:top_prob});
                inserted = true;
                break;
            }
        }
        if (!inserted && summary.tops[a][p].length < max_tops) {
            summary.tops[a][p].push({idx:sample.idx, prob:top_prob});
        }
        return {predicted:p, actual:a, prob:out_prob};
    };

    this.train = function(numTrain, callback) {
        var train_next_sample = function(t){            
            dataset.get_next_sample(t, function(sample){
                trainer.train(sample.x, sample.y);
                if (t+1 < numTrain) {
                    train_next_sample(t+1);
                } else {
                    callback();
                }
            });
        };
        check_if_ready(train_next_sample);
    };

    this.test = function(numTest, callback) {
        var results = [];
        var test_next_sample = function(t){
            dataset.get_next_sample(t, function(sample){
                result = test_sample(sample);
                results.push(result);
                if (t+1 < numTest) {
                    test_next_sample(t+1);
                } else {
                    callback(results);
                }
            });
        };
        check_if_ready(test_next_sample);
    };

    this.draw_sample = function(ctx, x_, y_, scale) {
        var V = net.layers[0].out_act;
        draw_volume(ctx, x_, y_, scale, V, 0, false);
    };

    this.draw_filter = function(ctx, layer, idx, x_, y_, scale) {
        var V = net.layers[layer].filters[idx];
        draw_volume(ctx, x_, y_, scale, V, idx, true);
    };

    this.draw_activations = function(ctx, layer, idx, x_, y_, scale) {
	    var V = net.layers[layer].out_act;
        draw_volume(ctx, x_, y_, scale, V, idx, false);
    };

    function draw_volume(ctx, x_, y_, scale, V, idx, is_weight) {
		var nx = V.sx;
        var ny = V.sy;
        var nz = V.depth;
		var nc = dataset.get_channels();
        if (nx * ny == 1) {
            nz = dataset.get_channels();
            nx = Math.sqrt(V.depth / nz);
            ny = nx;
        }
        var W = scale * nx;
        var H = scale * ny;
        var mm = maxmin(V.w);
        var img = ctx.createImageData(W, H);          
        for(var x=0; x<nx; x++) {
            for(var y=0; y<ny; y++) {
                var z = nz * (y * nx + x) + (is_weight ? 0 : idx);
                for(var dx=0; dx<scale; dx++) {
                    for(var dy=0; dy<scale; dy++) {
                        var idx_ = ((W * (y*scale+dy)) + (dx + x*scale)) * 4;
                        for (var c=0; c<3; c++) {
                            img.data[idx_+c] = Math.floor(255 * (V.w[z+c%nc] - mm.minv) / mm.dv);
                        }
                        img.data[idx_+3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(img, x_, y_);
    };

    function initialize() {
        var n = dataset.get_classes().length;
        summary = {
            confusion: [...Array(n).keys()].map(i => [...Array(n).keys()].map(i => 0)), 
            tops: [...Array(n).keys()].map(i => [...Array(n).keys()].map(i => [])), 
            actuals: [...Array(n).keys()].map(i => 0), 
            predictions: [...Array(n).keys()].map(i => 0), 
            correct: 0, total: 0
        };
        layer_defs = [];
        layer_defs.push({
            type:'input', 
            out_sx:dataset.get_dim(), 
            out_sy:dataset.get_dim(), 
            out_depth:dataset.get_channels()
        });
        idxSample = -1;
    };
    
    // initialize    
    var self = this;
    var maxmin = cnnutil.maxmin;
    var f2t = cnnutil.f2t;
    var dataset = dataset_;
    var max_tops = 64;
    var layer_defs;
    var trainer;
    var net;
    var summary;
    
    initialize();    
};
