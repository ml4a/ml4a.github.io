function demo(parent, width, height, datasetName_, numRows_, numCols_, scale_, margin_, classIdx_) 
{
	// canvas
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	// parameters
	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	var numRows = (numRows_ === undefined) ? 4 : numRows_;
	var numCols = (numCols_ === undefined) ? 8 : numCols_;
	var scale = (scale_ === undefined) ? 1.0 : scale_;
	var margin = (margin_ === undefined) ? 1 : margin_;
	var classIdx = (classIdx_ === undefined) ? null : classIdx_;

	// draw samples
	var data = new dataset(datasetName);
	data.load_labels(function(){
		data.load_batch(0, function(){
			data.draw_sample_grid(ctx, numRows, numCols, scale, margin, classIdx);
		});
	});
};