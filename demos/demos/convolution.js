// LABEL IMAGE PATCH AND FILTER
// CLEAN UP SETTINGS
// 1) train N samples/preload  2) test M (fwd pass demo, conv demo
// 2) train + test intermittently  (weights demo)



function demo(parent, width, height, datasetName_, drawAll_) 
{
	// canvas
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	// parametes
	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	var drawAll = (drawAll_ === undefined) ? false : drawAll_;

	// variables
	var net, data, dim, settings;
	var filter_size, pad_amt, num_filters, grid_size, sample_size;
	var crop_amt = 5;

	// display parameters for x1
    var rx1 = 330;
	var rx2 = 660;
	var ry1 = 10;
	var ry2 = 165;
	var ry3 = 40;
	
	// display parameters for all
	var fy = 240;
	var ay = 330;
	var margin = 60;

	// convnet parameters
	var layer = 1;
	var idx_filter = 1;
	var select_x = 0;
	var select_y = 0;
	var idx = 0;

	function draw_sample_as_grid(sample, x_, y_, cellsize) {
	    var nx = sample.sw;
	    var ny = sample.sh;
	    ctx.save();
	    ctx.translate(x_, y_);  
	    ctx.beginPath();
	    ctx.rect(0, 0, nx * cellsize.x, ny * cellsize.y);
	    ctx.strokeStyle = 'argb(255,0,0,1.0)';
	    ctx.lineJoin = ctx.lineCap = 'round';
	    ctx.lineWidth = 0.5;
	    ctx.stroke();
	    ctx.closePath();
	    for (var y=0; y<ny; y++) {
	        var ty = y * cellsize.y;
	        for (var x=0; x<nx; x++) {
	            var tx = x * cellsize.x;
	            var idx_color = 4*(x+y*nx);
	            ctx.save();
	            ctx.font = (cellsize.x/2.0)+'px Arial';
	            ctx.textAlign = 'center';   
	            ctx.textBaseline = 'middle';
	            ctx.translate(tx+cellsize.x/2.0, ty+cellsize.y/2.0);
	            ctx.fillText(sample.data[idx_color], 0, 0);
	            ctx.restore();
	        }
	    }
	    ctx.restore();
	};

	function loadPresetNetwork(callback) {		
		var snapshot;
		if (datasetName == 'MNIST') {
			snapshot = '/demos/datasets/mnist/mnist_snapshot_convnet.json';
		} else if (datasetName == 'CIFAR') {
			snapshot = '/demos/datasets/cifar/cifar10_snapshot_convnet.json';
		}
		data = new dataset(datasetName);
	    net = new convnet(data);
	    dim = data.get_dim();
		net.load_from_json(snapshot, callback);
	};

	function finished_training() {
	    net.test(idx, idx+1, finished_testing);
	};

	function finished_testing(results) {
	    sample = data.get_sample_image(idx, draw);
	};

	function draw_line(x1, y1, x2, y2, thickness, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.closePath();
    };

    function draw_bezier(x1, y1, x2, y2, x3, y3, x4, y4, thickness, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.moveTo(x1,y1);
        ctx.bezierCurveTo(x2,y2,x3,y3,x4,y4);
        ctx.stroke();
        ctx.closePath();
    };

    function draw_square(x1, y1, x2, y2, thickness, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.closePath();
    };

	function draw_square(x1, y1, x2, y2, thickness, color) {
		ctx.beginPath();
		ctx.strokeStyle = color;
    	ctx.lineWidth = thickness;
    	ctx.moveTo(x1, y1);
    	ctx.lineTo(x2, y1);
    	ctx.lineTo(x2, y2);
    	ctx.lineTo(x1, y2);
    	ctx.lineTo(x1, y1);
    	ctx.stroke();
    	ctx.closePath();
	};

	function draw_convolution_all() {
		var cell_size = settings.sample_scale + settings.sample_grid;
        var x1 = settings.sample_x + cell_size * select_x;
        var y1 = settings.sample_y + cell_size * select_y;
        var x2 = x1 + cell_size * filter_size;
        var y2 = y1 + cell_size * filter_size;

        // white background
        ctx.fillStyle = 'rgba(255,255,255,1.0)';
        ctx.fillRect(0, 0, width, height);

        // draw each filter
        for (var idx_filter=0; idx_filter<num_filters; idx_filter++) {
            var fw = 5 * (settings.s2 + 1);
            var aw = data.get_dim() * settings.s1;
            var fx = margin + (width-2*margin) * (idx_filter / (num_filters-1)) - fw*0.5;
            var ax = margin + (width-2*margin) * (idx_filter / (num_filters-1)) - aw*0.5;
            var bx1 = settings.sample_x + 0.5 * ((data.get_dim() + 2*pad_amt) * (settings.sample_scale + settings.sample_grid));
            var by1 = settings.sample_y + 1.0 * ((data.get_dim() + 2*pad_amt) * (settings.sample_scale + settings.sample_grid));

            // draw lines            
            draw_bezier(bx1, by1-2, bx1, by1 + 0.5 * (fy-by1), ax+aw/2, by1 + 0.5 * (fy-by1), ax+aw/2, fy, settings.sample_thickness, 'rgba(0,0,0,1.0)');
            draw_line(ax+aw/2, fy+fw/2, ax+aw/2, ay+aw/2, settings.sample_thickness, 'rgba(0,0,0,1.0)');

            // filter
            net.draw_filter(ctx, layer, idx_filter, fx, fy, settings.s2, 1);
            draw_square(fx, fy, fx+(settings.s2+1)*5, fy+(settings.s2+1)*5, settings.sample_thickness, 'rgba(0,0,0,1.0)');
            
            // activations
            net.draw_activations(ctx, layer, idx_filter, ax, ay, settings.s1);
            draw_square(ax + settings.s1*select_x, ay + settings.s1*select_y, ax + settings.s1*(select_x+1), ay + settings.s1*(select_y+1), 1.0, 'rgba(255,0,0,1.0)');
        }

        // current sample plus green square
        data.draw_current_sample(ctx, settings.sample_x, settings.sample_y, settings.sample_scale, settings.sample_grid, {x:0, y:0, w:data.get_dim()+2*pad_amt, h:data.get_dim()+2*pad_amt, pad:pad_amt});
        draw_square(x1, y1, x2, y2, settings.sample_thickness, 'rgba(0,255,0,1.0)');
	};

	function draw_convolution_x1() {
    	var cell_size = settings.sample_scale + settings.sample_grid;
    	var x1 = settings.sample_x + cell_size * select_x;
    	var y1 = settings.sample_y + cell_size * select_y;
    	var x2 = x1 + cell_size * filter_size;
    	var y2 = y1 + cell_size * filter_size;

    	// white background
    	ctx.fillStyle = 'rgba(255,255,255,1.0)';
    	ctx.fillRect(0, 0, width, height);

    	// current sample plus green square
		data.draw_sample(ctx, idx, settings.sample_x, settings.sample_y, settings.sample_scale, settings.sample_grid, {x:0, y:0, w:data.get_dim()+2*pad_amt, h:data.get_dim()+2*pad_amt, pad:pad_amt});
		draw_square(x1, y1, x2, y2, settings.sample_thickness, 'rgba(0,255,0,1.0)');
		
		// subsample
		data.draw_sample(ctx, idx, rx1, ry1+3, settings.s2, 1, {x:select_x, y:select_y, w:filter_size, h:filter_size, pad:pad_amt});
		draw_square(rx1, ry1+3, rx1 + (settings.s2+1)*filter_size, ry1+3 + (settings.s2+1)*filter_size, settings.sample_thickness, 'rgba(0,255,0,1.0)');
		
		// filter
		net.draw_filter(ctx, layer, idx_filter, rx1, ry2, settings.s2, 1);
		draw_square(rx1, ry2, rx1+(settings.s2+1)*5, ry2+(settings.s2+1)*5, settings.sample_thickness, 'rgba(0,0,0,1.0)');
		ctx.drawImage(summation_img, rx1+10+5*(settings.s2+1), ry1+55, 150, 180);
		
		// activations
		net.draw_activations(ctx, layer, idx_filter, rx2, ry3, settings.s1);
		draw_square(rx2 + settings.s1*select_x, ry3 + settings.s1*select_y, rx2 + settings.s1*(select_x+1), ry3 + settings.s1*(select_y+1), 1.0, 'rgba(255,0,0,1.0)');

		// selected activation
		var act_color = ctx.getImageData(rx2 + settings.s1*select_x + 1, ry3 + settings.s1*select_y + 1, 1, 1).data; 
		ctx.fillStyle = 'rgba('+act_color[0]+','+act_color[1]+','+act_color[2]+',1.0)';
		ctx.fillRect(rx1+160+5*(settings.s2+1), ry1+130, 30, 30);
		draw_square(rx1+160+5*(settings.s2+1), ry1+130, rx1+160+5*(settings.s2+1)+30, ry1+130+30, 1, 'rgba(0,0,0,1.0)');
	};

	function draw() {
		if (drawAll) {
			draw_convolution_all();
		} else {
			draw_convolution_x1();
		}
	};

	function toggle_draw_mode() {
		drawAll = !drawAll;
		set_sample_settings();
		draw();
	};

	function mouseMoved(evt) {
		function inside(mx, my, x, y, w, h) {
			return (mx > 0 && my > 0 && mx < w && my < h);
		};
		var canvas_rect = canvas.getBoundingClientRect();
		var mouse_x = evt.clientX - canvas_rect.left;
	    var mouse_y = evt.clientY - canvas_rect.top;
	    var smx = mouse_x - settings.sample_x; 
		var smy = mouse_y - settings.sample_y; 
		if (inside(smx, smy, 0, 0, (dim + 2*pad_amt - filter_size+1) * (settings.sample_scale + settings.sample_grid), (dim + 2*pad_amt - filter_size+1) * (settings.sample_scale + settings.sample_grid))) {
			console.log(pad_amt, dim, settings.sample_grid, settings.sample_scale);
			select_x = Math.min(dim-1, Math.floor(smx / (settings.sample_scale + settings.sample_grid)));
			select_y = Math.min(dim-1, Math.floor(smy / (settings.sample_scale + settings.sample_grid)));
			draw();
		} 
	};

	function set_sample_settings() {
		if (drawAll) {
		    settings = {
	            sample_x: 0.5 * width - 0.5 * ((data.get_dim() + 2 * pad_amt) * 5),
	            sample_y: 10,
	            sample_scale: 4,
	            sample_grid: 1,
	            sample_thickness: 2,
				s1: 4.0,
			    s2: 12.0
	        };
	    } else {
	    	settings = {
	            sample_x: 12,
	    		sample_y: 12,
	    		sample_scale: 9,
	    		sample_grid: 1,
	    		sample_thickness: 4,
	           	s1: 9	,
				s2: 24
	        };
	    }
	};
	
	function finished_loading() {
		pad_amt = 0;//net.get_net().layers[1].pad;
		filter_size = net.get_net().layers[1].filters[0].sx;
	    num_filters = net.get_net().layers[1].filters.length;
	    grid_size = data.get_dim() + 2 * (pad_amt - crop_amt) - filter_size + 1;
	    sample_size = data.get_dim() + 2 * (pad_amt - crop_amt);
	    set_sample_settings();
		net.test(0, 1, function() {
			canvas.addEventListener("mousemove", mouseMoved, false);
			finished_testing();
		});
	};

	// assets loaded, start demo
	function ready() {
		loadPresetNetwork(finished_loading);
	};

	function next_sample() {
		idx+=1;
		net.test(idx, idx+1, finished_testing);
	};

	function next_filter() {
		idx_filter = (idx_filter + 1) % num_filters;
		draw();
	};

	// control pannel
	set_control_panel_height(parent.description_panel_div, 70);
	set_text_panel(parent.description_panel_div, 'This demo shows how convolution works in a convolutional layer. A filter is slid along every horizontal and vertical position of the original image or the previous layer\'s activations, and the dot product is taken in each position. The resulting activation map (on the right) shows the presence of the feature map -- or roughly patterns in the input which resemble the filter itself. Move your mouse around the input to see individual patches, and click \'next sample\' or \'next filter\' to see different convolutional filters and inputs in action.');
	//set_text_panel(parent.description_panel_div, 'A visualization of the set of filters in the first convolutional layer, and their corresponding activation maps going to the next layer.'); 
	add_control_panel_action('next_sample', next_sample);
	add_control_panel_action('next_filter', next_filter);
	add_control_panel_action('draw all?', toggle_draw_mode);

	// load assets
	var summation_img = new Image();
    summation_img.onload = ready;
	summation_img.src = '/images/summation2.png';
};
