
var trainChart = dc.seriesChart("#train-chart"),
    valChart = dc.seriesChart("#val-chart"),
    modelChart = dc.rowChart("#model-chart"),
    modeltypeChart  = dc.rowChart("#chart-modeltype"),
    filterChart = dc.rowChart("#chart-filters"),
    lrRegChart = dc.heatMap("#heatmap");


d3.json("data.json", function(error, data) {
    console.log(data); // this is your data
    data = flattenModels(data);
    console.log(data);
	var ndx = crossfilter(data);

    // First plot: iterations
    var runDimension1 = ndx.dimension(function(d) {return [+d.model, +d.iteration]; });
    var runDimension2 = ndx.dimension(function(d) {return [+d.model, +d.iteration]; });
    //var runGroup = runDimension.group();
    var runValAcc = runDimension1.group().reduceSum(function(d) { return +d.val_acc; });
    var runTrainAcc = runDimension2.group().reduceSum(function(d) { return +d.train_acc; });

    //Second plot: select model
    var modelDimension = ndx.dimension(function(d) {return +d.model; });
    var modelAccGroup = reductio().max(function(d) {return +d.final_val_acc;})(modelDimension.group());

    //Third plot: modeltype
    var modeltypeDim = ndx.dimension(function(d){return d.modeltype;});
    var modelTypeGroup = modeltypeDim.group();
    var countPerModeltype = reductio()
                            .exception(function(d) {return d.model;})
                            .exceptionCount(true)(modelTypeGroup);
    //var accPerModeltype = modelTypeGroup.reduceSum(function(d) {return d.final_val_acc;});

    //Fourth plot: Nr of  conv layers
    var nrconvlayersDim = ndx.dimension(function(d) {return +d.nr_convlayers;});
    var convLayerGroup = nrconvlayersDim.group();
    var accPerConvlayer = reductio()
                            .exception(function(d) {return d.model;})
                            .exceptionCount(true)
                            .exceptionSum(function(d) {return d.final_val_acc;})
                            (nrconvlayersDim.group());

    // Fifth plot: Learning rates/Regularization rate bubble chart
    var lrRegDim = ndx.dimension(function(d){return [+Math.log(d.learning_rate)/Math.log(10), +Math.log(d.regularization_rate)/Math.log(10)];});
    var lrRegGroup = lrRegDim.group(function(g){return [Math.round(g[0]), Math.round(g[1])];});
    var avgAccHeatmap = reductio()
                            .exception(function(d) {return d.model;})
                            .exceptionCount(true)
                            .exceptionSum(function(d) {return d.final_val_acc;})
                            (lrRegGroup);

    //console.log(accPerConvlayer.all());
    valChart
      .chart(dc.lineChart)
      .x(d3.scale.linear())
      .brushOn(false)
      .yAxisLabel("Validation accuracy")
      .xAxisLabel("Iteration")
      .colors(d3.scale.category20())
      .elasticX(true)
      .dimension(runDimension1)
      .group(runValAcc)
      .seriesAccessor(function(d) {return "Model " + d.key[0];})
      .keyAccessor(function(d) {return +d.key[1];})
      .valueAccessor(function(d) {return +d.value;})
      .controlsUseVisibility(true);

  trainChart
    .chart(dc.lineChart)
    .x(d3.scale.linear())
    .brushOn(false)
    .yAxisLabel("Train accuracy")
    .xAxisLabel("Iteration")
    .colors(d3.scale.category20())
    .elasticX(true)
    .dimension(runDimension2)
    .group(runTrainAcc)
    .seriesAccessor(function(d) {return "Model " + d.key[0];})
    .keyAccessor(function(d) {return +d.key[1];})
    .valueAccessor(function(d) {return +d.value;})
    .controlsUseVisibility(true);

  modelChart
    .margins({top: 0, left: 10, right: 10, bottom: 20})
      .dimension(modelDimension)
      .group(modelAccGroup)
       .valueAccessor(function(d) {return +d.value.max;})
      .elasticX(true)
      .colors(d3.scale.category20()) // Use the same colors as the valChart
      .controlsUseVisibility(true);

    modeltypeChart
      .margins({top: 0, left: 10, right: 10, bottom: 20})
        .dimension(modeltypeDim)
        .group(countPerModeltype)
         .valueAccessor(function(d) {return +d.value.exceptionCount;})
        .elasticX(true)
        .controlsUseVisibility(true);

	filterChart
    .margins({top: 0, left: 10, right: 10, bottom: 20})
      .dimension(nrconvlayersDim)
      .group(accPerConvlayer)
      .valueAccessor(function(d) {
          if(d.value.exceptionCount === 0){return 0;}
          else {return +(d.value.exceptionSum / d.value.exceptionCount);}
      })
      .label(function(d) {
          if(d.key == 1) { return d.key + " layer";}
          else { return d.key + " layers"; }
      })
      .elasticX(false)
      .controlsUseVisibility(true);

      var heatColorMapping = function(d) {
          var t = 0.75;
           if (d < t) {
              return d3.scale.linear().domain([0,t]).range(["red", "yellow"])(d);
          }
          else {
              return d3.scale.linear().domain([t,1]).range(["yellow", "green"])(d);
          }
        return d3.scaleSequential(d3.interpolateInferno)(d);
      };
      heatColorMapping.domain = function() {
          return [0, 1];
      };

      lrRegChart
                .width(400)
                .height(300)
                .dimension(lrRegDim)
                .group(avgAccHeatmap)
                .keyAccessor(function(d) { return +d.key[0]; })
                .valueAccessor(function(d) { return +d.key[1]; })
                .colorAccessor(function(d) {
                    return +(d.value.exceptionSum / d.value.exceptionCount);
                })
                .colsLabel(function(d){return "10^" + d;})
                .rowsLabel(function(d){return "10^" + d;})
                .title(function(d) {
                    return " Learning rate:   10^" + d.key[0] + "\n" +
                           "  Regularzation rate:   10^" + d.key[1] + "\n" +
                           "  Avg acc:   " + (d.value.exceptionSum / d.value.exceptionCount);})
                .colors(heatColorMapping)
                .calculateColorDomain();

	dc.renderAll();
});
