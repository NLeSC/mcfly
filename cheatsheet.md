# mcfly Cheatsheet

For detailed documentation see:
https://github.com/NLeSC/mcfly/wiki/Home---mcfly
For tutorials see:https://github.com/NLeSC/mcfly-tutorial

### Input data:
*X_train* =>  N samples X N timesteps X  N channels

*y_train* => N samples X N classes

### Generate models:

```
num_classes = y_train_binary.shape[1]
models = modelgen.generate_models(X_train.shape, number_of_classes=num_classes, number_of_models = 2)
```

### Train multiple models:
```
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(
  X_train, y_train_binary, X_val, y_val_binary,
  models,nr_epochs=5,subset_size=300,
  verbose=True, outputfile=outputfile)
```
### Select best model
```
best_model_index = np.argmax(val_accuracies)
best_model, best_params, best_model_types = models[best_model_index]
```

### Train one specific model (this is done with Keras function fit):
```
best_model.fit(X_train, y_train_binary,
  nb_epoch=25, validation_data=(X_val, y_val_binary))
```
