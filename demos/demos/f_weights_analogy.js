function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');


    function getSubscript(idx) {
        var subscripts = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
        var o = idx%10;
        var t = Math.floor(idx/10.0)%10;
        var h = Math.floor(idx/100.0)%100;
        return (idx<100?'':subscripts[h])+(h==0&&t<1?'':subscripts[t])+(subscripts[o]);
    };


    var draw_input_grid = function(x_, y_, cellsize, nx, ny, nvx, nvy, text) {
        var ellipsis = function(x, y, direction) {
            ctx.save();
            ctx.font = (cellsize.x/3.0)+'px Arial';
            ctx.textAlign = 'center';   
            ctx.textBaseline = 'middle';
            ctx.translate(cellsize.x * (x+0.5), cellsize.y * (y+0.5));
            ctx.fillText('.', 0, 0);
            if (direction == 0) {
                ctx.fillText('.', -cellsize.x/4.0, -cellsize.y/4.0);
                ctx.fillText('.', +cellsize.x/4.0, +cellsize.y/4.0);
            }
            else if (direction == 1) {
                ctx.fillText('.', 0, -cellsize.y/4.0);
                ctx.fillText('.', 0, +cellsize.y/4.0);
            }
            else if (direction == 2) {
                ctx.fillText('.', -cellsize.x/4.0, 0);
                ctx.fillText('.', +cellsize.x/4.0, 0);
            }
            ctx.restore();
        }
        var draw_cell = function(x, y, txt) {
            var tx = x * cellsize.x;
            var idx_color = 4*(x+y*nx);
            ctx.save();
            ctx.font = (cellsize.x/3.0)+'px Arial';
            ctx.textAlign = 'center';   
            ctx.textBaseline = 'middle';
            ctx.translate(tx+cellsize.x/2.0, ty+cellsize.y/2.0);
            ctx.fillText(txt, 0, 0);
            ctx.restore();
        }
        ctx.save();
        ctx.translate(x_, y_);  
        ctx.beginPath();
        ctx.rect(0, 0, (nvx + 1) * cellsize.x, (nvy + 1) * cellsize.y);
        ctx.strokeStyle = 'argb(255,0,0,1.0)';
        ctx.lineJoin = ctx.lineCap = 'round';
        ctx.lineWidth = 0.5;
        ctx.stroke();
        ctx.closePath();

        for (var y=0; y<nvy+1; y++) {
            var ty = y * cellsize.y;
            for (var x=0; x<nvx+1; x++) {
                if (x == nvx-1 && y == nvy-1) {
                    ellipsis(x, y, 0);
                }
                else if (y == nvy-1) {
                    if (x == 0) {
                        ellipsis(x, y, 1);
                    }
                }
                else if (x == nvx-1) {
                    if (y == 0) {
                        ellipsis(x, y, 2);
                    }
                }
                else {            
                    var idx = (x == nvx ? nx : x+1) + (y == nvy ? (ny-1) : y) * nx;
                    draw_cell(x, y, text+getSubscript(idx));
                }
            }
        }
        ctx.restore();
    };

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

    var settings_vis1 = {
        context: ctx,
        width: 420, 
        height: 320,
        architecture: [784,10],
        visible: [12,10],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 1,
            radius: 7,
        },
        connectionStyle: {
            color: 'rgba(50,50,50,0.9)',
            arrowLen: 0,
            thickness: 0.5
        }
    };

    var settings_vis2 = {
        context: ctx,
        width: 320, 
        height: 320,
        architecture: [784, 1],
        visible: [6, 1],
        heightBounds: [[0, 1], [0.07, 0.93]],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 1.5,
            radius: 15,
            ellipsisRadius: 2,
            ellipsisMargin: 6
        },
        connectionStyle: {
            color: 'rgba(40,40,40,0.5)',
            arrowLen: 0,
            arrowWidth: 4,
            labelText: 'W',
            labelLerp:0.2,
            thickness: 1
        }
    };



    var vis1 = new NetworkVisualization(settings_vis1);
    var vis2 = new NetworkVisualization(settings_vis2);

    vis1.setNeuronStyle({
        leftLabelText: "pixel",
        leftLabelCounter: true,
        leftLabelMargin: 30,
        leftLabelSize: 16}, 0);

    vis1.setNeuronStyle({radius:17}, 1);

    vis1.setConnectionStyle({
        thickness:0.25, 
        color:'rgba(0,0,0,0.6)'});

    vis1.setConnectionStyle({
        thickness:1, 
        color:'rgba(0,0,0,1.0)'}, 0, undefined, 0);

    vis2.setNeuronStyle({
        radius:22,
        labelText: 'X',
        labelCounter: true,
        labelSize: 18,
        labelCounterSubscript: true}, 0);

    vis2.setNeuronStyle({
        radius:50, 
        thickness:2.5,
        labelSize:30,
        labelText:'Z'}, 1);

    vis2.setConnectionStyle({
        labelText: 'W',
        labelCounter: true,
        labelCounterSubscript: true});


    vis1.draw(30, 0);
    vis2.draw(500, 0);

    draw_input_grid(20, 360, {x:60, y:48}, 28, 28, 4, 4, 'X');
    draw_input_grid(380, 360, {x:60, y:48}, 28, 28, 4, 4, 'W');


    ctx.font = '32px Arial';
    ctx.textAlign = 'center';   
    ctx.textBaseline = 'middle';        
    ctx.save();
    ctx.beginPath();
    ctx.arc(350, 480, 5, 0,2*Math.PI, false);
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.arc(780, 480, 50, 0,2*Math.PI, false);
    ctx.closePath();
    ctx.fillText('=', 705, 480)
    ctx.fillText('Z', 780, 480)
    ctx.stroke();
    ctx.restore();

};