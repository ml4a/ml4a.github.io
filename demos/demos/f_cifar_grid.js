var canvas = document.createElement("canvas");
canvas.width = 2200;
canvas.height = 600;
//document.getElementById("post").appendChild(canvas);
document.body.appendChild(canvas);

var ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = true;



var data = new dataset('CIFAR');
data.load_labels(function(){
	data.load_batch(0, function(){
		data.draw_sample_grid(ctx, 5, 30, 2, 2, null);
	});
});