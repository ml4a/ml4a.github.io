var demo = function(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');


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

    function arrow(ctx, x1, x2, y, arrowLen) {
        ctx.beginPath();
        ctx.lineWidth = 3.0;
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y);
        ctx.moveTo(x2, y);
        ctx.lineTo(x2-arrowLen, y-arrowLen);
        ctx.moveTo(x2, y);
        ctx.lineTo(x2-arrowLen, y+arrowLen);
        ctx.stroke();
        ctx.closePath();
    };

    var data = new dataset('MNIST');
    var idx_sample = Math.floor(Math.random()*1000);
    var y1 = 10;
    var y2 = 620;
    var x1 = 910;
    var x2 = 1010;

    var net = new NetworkVisualization({
        context: ctx,
        width: 240, 
        height: 610,
        architecture: [784, 5],
        visible: [20, 5],
        heightBounds: [[0, 1], [0.07, 0.93]],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 1.5,
            radius: 10,
            ellipsisRadius: 2,
            ellipsisMargin: 6
        },
        connectionStyle: {
            color: 'rgba(40,40,40,0.5)',
            arrowLen: 0,
            arrowWidth: 4,
            thickness: 1
        }
    });

    net.setNeuronStyle({
        leftLabelSize: 12,
        leftLabelText: 'pixel',
        leftLabelMargin: 20,
        leftLabelCounter: true}, 0);

    net.setNeuronStyle({
        radius:30, 
        thickness:2.5}, 1);

    data.get_sample_image(idx_sample, function(sample){
        data.draw_sample(ctx, idx_sample, 4, 230, 5, 1);

        ctx.textAlign='center';
        ctx.font='22px Arial';
        ctx.fillText('28 x 28', 4+28*6/2, 230+28*6+25);
        ctx.fillText('784 pixels', 4+28*6/2, 230+28*6+55);

        arrow(ctx, 175, 205, 314, 10)
        draw_sample_as_grid(sample, 208, 10, {x:25, y:22});
        net.draw(980, 10);
        var xm = (x1 + x2) / 2;
        var ym = (y1 + y2) / 2;
        bezier(ctx, x1, ym, xm, ym, x1, y1, xm, y1);
        bezier(ctx, x1, ym, xm, ym, x1, y2, xm, y2);
    });

}