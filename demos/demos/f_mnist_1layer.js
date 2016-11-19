function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');


    var draw_sample_as_grid = function(sample, x_, y_, cellsize) {
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

    function bezier(ctx, x1, y1, x2, y2, x3, y3, x4, y4) {
        ctx.beginPath();
        ctx.lineWidth = 3.0;
        ctx.moveTo(x1,y1);
        ctx.bezierCurveTo(
            x2,y2,
            x3,y3,
            x4,y4);
        ctx.stroke();
        ctx.closePath();
    };

    var settings = {
        context: ctx,
        width: 420, 
        height: 360,
        architecture: [784,10],
        visible: [21,10],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 1,
            radius: 7,
        },
        connectionStyle: {
            color: 'rgba(50,50,50,0.9)',
            arrowLen: 0,
            thickness: 0.5,

        }
    };


    var y1 = 5;
    var y2 = settings.height-5;
    var x1 = 175;
    var x2 = 258;


    var data = new dataset('MNIST');
    var net = new NetworkVisualization(settings);

    net.setNeuronStyle({
        leftLabelText: "pixel",
        leftLabelCounter: true,
        leftLabelMargin: 30,
        leftLabelSize: 16}, 0);
    net.setNeuronStyle({radius:17}, 1);


    data.load_next_batch(function(){
        var idx_sample = 299;    
        data.draw_sample(ctx, idx_sample, 5, 80, 5, 1);    
        net.draw(260, 0);
        var xm = (x1 + x2) / 2;
        var ym = (y1 + y2) / 2;
        bezier(ctx, x1, ym, xm, ym, x1, y1, xm, y1);
        bezier(ctx, x1, ym, xm, ym, x1, y2, xm, y2);
    });

};