var flattenModels = function(obj){
    var output = [];
    for(var modelNr = 0; modelNr<obj.length; modelNr++) {
        model = obj[modelNr];

        let val_metric = model.metrics.val_accuracy;
        let train_metric = model.metrics.accuracy;
        for(var iteration=0; iteration<val_metric.length; iteration++){
            modeldict = {};

            //Model identifier
            modeldict.model = modelNr;
            //Model specific arguments
            modeldict.modeltype = model.modeltype;
            modeldict.learning_rate = model.learning_rate;
            modeldict.regularization_rate = model.regularization_rate;

            // Calculate number of layers:
            if(model.modeltype==='CNN'){
              modeldict.nr_layers = model.filters.length;
            }
            if(model.modeltype==='DeepConvLSTM'){
              modeldict.nr_layers = model.filters.length + model.lstm_dims.length;
            }
            if(model.modeltype==='ResNet'){
              modeldict.nr_layers = model.network_depth*3;
            }
            if(model.modeltype==='InceptionTime'){
              modeldict.nr_layers = model.network_depth*2;
            }

            modeldict.final_val_acc = val_metric[val_metric.length-1];

            //Iteration specific arguments
            modeldict.iteration = iteration;
            modeldict.val_metric = val_metric[iteration];
            modeldict.train_metric = train_metric[iteration];
            modeldict.val_loss = model.metrics.val_loss[iteration];
            modeldict.train_loss = model.metrics.loss[iteration];

            output.push(modeldict);
        }
    }
    return output;
};
