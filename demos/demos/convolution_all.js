function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');



    var settings = {
        context: ctx,
        rect: {x:0, y:0, w:600, h:400},
        architecture: [3, 2, 1],
        visible: [3, 2, 1],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 4,
            radius: 45,
            labelSize: 35,
        },
        connectionStyle: {
            color: 'rgba(40,40,40,0.5)',
            arrowLen: 20,
            arrowWidth: 5,
            thickness: 3
        }
    };


    var net = new NetworkVisualization(settings);
    net.setHeightBounds([0.07,0.93],1);

    var idx = 0;

    var step1 = function(){
        net.setNeuronStyle({
            biasLabelText: 'b=0.08',
            biasLabelSize: 16,
            labelText: '0.08'}, 1);
        redraw();
    };

    var step2 = function(){
        net.setConnectionStyle({
            labelText: 'w = 0.08',
            labelSize: 15,
            labelLerp: 0.25}, 1, 1);
        redraw();
    };

    var step3 = function(){
        net.setConnectionStyle({
            color: 'rgba(0,0,0,0.9)',
            thickness: 4}, 1, 0);
        redraw();
    };

    var next = function() {
        idx = (idx + 1) % 3;
        if (idx == 1) step1();
        if (idx == 2) step2();
        if (idx == 3) step3();
        redraw();
    };



    var redraw = function() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        net.draw();
    }

    window.addEventListener("keydown", function(e) { 
        if (e.keyCode == 49) {
            next();
        } 
    }, false);

};