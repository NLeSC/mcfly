var flattenModels = function(obj){
    var output = [];
    for(var modelNr = 0; modelNr<obj.length; modelNr++) {
        model = obj[modelNr];
        for(var iteration=0; iteration<model.val_acc.length; iteration++){
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

            modeldict.final_val_acc = model.val_acc[model.val_acc.length-1];

            //Iteration specific arguments
            modeldict.iteration = iteration;
            modeldict.val_acc = model.val_acc[iteration];
            modeldict.train_acc = model.train_acc[iteration];
            modeldict.val_loss = model.val_loss[iteration];
            modeldict.train_loss = model.train_loss[iteration];

            output.push(modeldict);
        }
    }
    return output;
};
