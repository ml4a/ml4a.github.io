function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');


    var settings = {
        context: ctx,
        width: 420, 
        height: 300,
        architecture: [3, 1, 1],
        visible: [3, 1, 1],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 4,
            radius: 40,
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
    net.setNeuronStyle({radius:60}, 1);
    net.setNeuronStyle({radius:0}, 2);
    net.setNeuronStyle({labelText:"X₁"}, 0, 0);
    net.setNeuronStyle({labelText:"X₂"}, 0, 1);
    net.setNeuronStyle({labelText:"X₃"}, 0, 2);
    net.setNeuronStyle({biasLabelSize:36, biasLabelText:"b"}, 1, 0);
    net.draw(0, 0);

    ctx.save();
    ctx.translate(settings.width + 16, 0.5 * settings.height - 5);
    ctx.font = '54px Arial';
    ctx.textAlign = 'center';   
    ctx.textBaseline = 'middle';        
    ctx.fillText("y", 0, 0);
    ctx.restore();

};