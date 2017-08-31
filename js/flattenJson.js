var flattenModels = function(obj){
    var output = [];
    for(var modelNr = 0; modelNr<obj.length; modelNr++) {
        model = obj[modelNr];
        // For backwards compatibility with data generated with version <=1.0.1
        let val_metric = model.val_metric ? model.val_metric : model.val_acc;
        let train_metric = model.train_metric? model.train_metric : model.train_acc;
        for(var iteration=0; iteration<val_metric.length; iteration++){
            modeldict = {};

            //Model identifier
            modeldict.model = modelNr;
            //Model specific arguments
            modeldict.modeltype = model.modeltype;
            modeldict.learning_rate = model.learning_rate;
            modeldict.regularization_rate = model.regularization_rate;
            modeldict.nr_convlayers = model.filters.length;
            if(model.lstm_dims===undefined){
                modeldict.nr_lstmlayers = 0;
            }
            else {
                modeldict.nr_lstmlayers = model.lstm_dims.length;
            }

            modeldict.final_val_acc = val_metric[val_metric.length-1];

            //Iteration specific arguments
            modeldict.iteration = iteration;
            modeldict.val_metric = val_metric[iteration];
            modeldict.train_metric = train_metric[iteration];
            modeldict.val_loss = model.val_loss[iteration];
            modeldict.train_loss = model.train_loss[iteration];

            output.push(modeldict);
        }
    }
    return output;
};
