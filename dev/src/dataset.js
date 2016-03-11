function dataset() 
{
	var dim;
	var channels;
	var classes;
	var rows_per_batch;
	var num_batches;
	var callback;
	var training_set=[];

	this.get_training_sample = function(t) {
		return training_set[t];
	};

	this.get_test_sample = function(t) {
		return training_set[t];
	};

	this.get_dim = function() {
		return dim;
	};

	this.get_channels = function() {
		return channels;
	};

	this.get_classes = function() {
		return classes;
	};

	this.loadMNIST = function(callback_) {
		callback = callback_;
		root_dir = '/datasets/mnist/mnist';
		dim = 28;
		channels = 1;
		rows_per_batch = 3000;
		num_batches = 4; //20
		classes = ["0","1","2","3","4","5","6","7","8","9"];
		for (var i=0; i<num_batches; i++) { 
			this.load_batch(i);
		}
	};

	this.loadCIFAR = function(callback_) {
		callback = callback_;
		root_dir = '/datasets/cifar/cifar10';
		dim = 32;
		channels = 3;
		rows_per_batch = 1000;
		num_batches = 10; //50;
		classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];
		for (var i=0; i<num_batches; i++) { 
			this.load_batch(i);
		}
	};

	this.load_batch = function(batch_idx) {
		var batch_path = root_dir+"_batch_"+batch_idx+".png";
		loadImage(batch_path, function(img) {
			var w = dim;
			var nc = channels;
			var n = rows_per_batch;
			img.loadPixels();
			for (var r=0; r<n; r++) {
	    	var b_label = labels[n * batch_idx + r];
	    	var b_vol = new convnetjs.Vol(w, w, nc, 0.0);
	    	var W = w * w;
	    	for (var i=0; i<W; i++) {
	     		var ix = ((W * r) + i) * 4;
	      	for (var c=0; c<nc; c++) {
		      	b_vol.w[nc*i+c] = img.pixels[ix+c] / 255.0; 
					}	
	    	}
	    	var sample = {vol:b_vol, label:b_label};
	    	training_set.push(sample);
			}
			console.log("training set size: " +training_set.length);
			if (training_set.length == num_batches * rows_per_batch && callback != null) {
				callback();
			}
		});
	};

	this.get_image = function(sample) {
		var nc = channels;
		var img = createImage(dim, dim);
		img.loadPixels();
		for (var i=0; i<dim*dim; i++) {
			for (var j=0; j<3; j++) {	
				var iw = nc * i + (nc == 1 ? 0 : j);
				img.pixels[4*i+j] = 255 * sample.vol.w[iw];
			}
			img.pixels[4*i+3] = 255;
		}
		img.updatePixels();
		return img;
	};
};