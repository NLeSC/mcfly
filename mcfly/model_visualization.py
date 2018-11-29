"""
Copyright (C) 2018 by Tudor Gheorghiu
Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated
documentation files (the "Software"),
to deal in the Software without restriction,
including without l> imitation the rights to
use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
"""
import matplotlib.pyplot as plt
import numpy as np
import viznet
import keras



# --------------------------------
# Visualization helper functions:
# --------------------------------
    
def draw_box(pos_x, pos_y, size, face_color, text0, text1=None):
    # draw textbox using matplotlib
    if text1 is None:
        text = text0
    else:
        #text = "  " + "\n" + text1
        plt.text(pos_x, pos_y, text1, size=0.8*size, rotation=0,
             horizontalalignment="center", 
             verticalalignment="bottom")
        text = text0 + "\n" + "  "
        
    plt.text(pos_x, pos_y, text, size=size, rotation=0,
             horizontalalignment="center", 
             verticalalignment="bottom",
             bbox=dict(boxstyle="round",
                       ec=(0, 0, 0, 0.5),
                       fc=face_color))


def draw_arrow(pos_x, pos_y, dx, dy, width):
    #plt.annotate("", xy=(pos_x, pos_y), xytext=(pos_x, pos_y-dy), arrowprops=dict(arrowstyle="->"))

    plt.arrow(pos_x, pos_y, -0.2*dx, -0.2*dy, width=width)

    
def draw_model_header(pos_x, pos_y, scale_y, model_no, model_type, size):
    # header: model number and type

    header_dy = size/500 * scale_y
    plt.text(pos_x, pos_y, "Model "+str(model_no), size=1.5*size, rotation=0,
             horizontalalignment="center", verticalalignment="center")
    plt.text(pos_x, pos_y - header_dy, "type: "+model_type, size=size, rotation=0,
             horizontalalignment="center", verticalalignment="center")


def get_spaced_colors(n, alpha):
    # generate a list of different colors (as RGBA format) with given alpha

    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    RGBA_colors = [(int(i[1::3], 16)/255, int(i[0::3], 16)/255, int(i[2::3], 16)/255, alpha) for i in colors]    

    return RGBA_colors



def collect_layer_infos(models):
    # Collect infos of model layers (relevant for plotting)
    
    model_types = []  # type of model (CNN, LSTM...)
    model_layers = []  # layer types
    model_layer_infos = []  # additional layer infos (such as no. of filters)

    for model_id, item in enumerate(models):
        model, params, model_type = item
        
        if model_type == 'DeepConvLSTM':
            model_types.append('LSTM')
        else:
            model_types.append(model_type)
            
        layer_types = []
        layer_infos = []

        # Read layer types
        for m, layer in enumerate(model.layers):
            if (type(layer) == keras.layers.convolutional.Conv2D):
                layer_types.append("Conv2D");
                layer_infos.append("Filters: " + str(layer.get_config()["filters"]))
            elif (type(layer) == keras.layers.convolutional.Conv1D):
                layer_types.append("Conv1D");
                layer_infos.append(str(layer.get_config()["filters"]) + " filters")
            elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                layer_types.append("MaxPooling2D");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Dropout):
                layer_types.append("Dropout");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Flatten):
                layer_types.append("Flatten");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Activation):
                layer_types.append(layer.get_config()["activation"])
                layer_infos.append("")
            elif (type(layer) == keras.layers.normalization.BatchNormalization):
                layer_types.append("BatchNorm");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Dense):
                layer_types.append("Dense");
                layer_infos.append("Units: "+ str(layer.get_config()["units"]))
            elif (type(layer) == keras.layers.recurrent.LSTM):
                layer_types.append("LSTM");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Reshape):
                layer_types.append("Reshape");
                layer_infos.append("")
            elif (type(layer) == keras.layers.wrappers.TimeDistributed):
                layer_types.append("TimeDistributed");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Lambda):
                layer_types.append("Lambda");
                layer_infos.append("")
            else:
                layer_types.append(layer)

        model_layers.append(layer_types)
        model_layer_infos.append(layer_infos)

    return model_types, model_layers, model_layer_infos



def model_overview(models, 
                   max_num_models=8, 
                   scale_grid = 1,
                   scale_box=2, 
                   extra_layer_info=True):
    ''' 
    Visual comparison of different deep learning models
    
    max_num_models -- maximum number of models that should be displayed in one plot
    extra_layer_info -- if True-> show additional layer information (such as no. of filters)
    '''

    # Collect relevant layer information from all models
    model_types, model_layers, model_layer_infos = collect_layer_infos(models) 
    
    # Find all unique layer types
    layers_all = []
    for layer in model_layers:
        for x in layer:
            layers_all.append(x)
    layers_unique = list(set(layers_all)) 

    # Access dimensions:
    num_models = min(len(models), max_num_models)  # number of models to compare 
    maxlength_models = np.max([len(x) for x in model_layers])
    
    # Set plot dimensions
    scale_x = scale_grid * num_models
    scale_y = scale_x
    plot_scaling = min(1, 5/num_models)  # scale plot when comparing many models
    box_size = scale_box*scale_y*plot_scaling

    dx = scale_x/(num_models + 1)
    dy = plot_scaling * 0.06 * scale_y
    header_y = scale_y * (0.05 + plot_scaling * 0.1)

    size_y = header_y + (maxlength_models+5)*dy
    
    #plt.rcParams['figure.figsize'] = [scale_x, 1.25*max(size_y, scale_y)]
    fig = plt.figure(figsize=(scale_x, 1.25*max(size_y, scale_y)))
    #fig, ax = plt.subplots()
#    plt.ylim(0, size_y) #(1- (header_y + maxlength_models*dy), 1)

    # Choose colors for plot
    selected_colors = get_spaced_colors(len(layers_unique)+4, alpha=0.2) 

    # Plot layers of all models
    for i in range(num_models):
        pos_x = dx/2 + i * dx
        pos_y = size_y - 0.05*scale_y
        draw_model_header(pos_x, pos_y, scale_y, i+1, model_types[i], box_size)
        
        #initialize pos_y
        pos_y = size_y - header_y + dy
        for j in range(len(model_layers[i])):
            color = selected_colors[layers_unique.index(model_layers[i][j])]
            
            # check if extra information present
            if model_layer_infos[i][j] == "":
                pos_y = pos_y - dy
                draw_box(pos_x, pos_y, box_size, color, model_layers[i][j])
                if j > 0:
                    draw_arrow(pos_x, pos_y+0.8*dy, 0, dy, 0.01*scale_grid)
            else:
                pos_y = pos_y - 1.5*dy
                draw_box(pos_x, pos_y, box_size, color, model_layers[i][j], text1=model_layer_infos[i][j])  
                if j > 0:
                    draw_arrow(pos_x, pos_y+0.8*1.5*dy, 0, dy, 0.01*scale_grid)
    
    #
    print(dx, dy, size_y, plot_scaling, pos_y)
    plt.axis('equal')
    plt.xlim(-dx, scale_x)
    print("max ", max(size_y, scale_y))
    plt.ylim(min(0, pos_y), 1.1*max(size_y, scale_y))
    #ax.set_xlim(0, scale_x)
    #ax.set_xlim(0, scale_x)
    #ax.set_ylim(0, max(size_y, scale_y))
    #plt.ylim(0, max(size_y - pos_y, size_y))
    plt.axis('off')
    plt.savefig("test.png")
    plt.show()
    plt.savefig("test1.png", bbox_inches='tight')
    #print(ax.get_ylim())
    plt.savefig("fig_test.pdf", bbox_inches='tight')
    
    
    

def modelcompare(models):
    # visual comparison of different deep learning models

    num_models = len(models)
    
    model_layers = []
    model_layer_infos = []
    #model_configs = []
    #model_in_out = []
    for model_id, item in enumerate(models):
        model, params, model_types = item
    
        layer_types = []
        layer_infos = []
        layer_config = []
        layer_in_out = []
        # Read layer types
        for m, layer in enumerate(model.layers):
            if (type(layer) == keras.layers.convolutional.Conv2D):
                layer_types.append("Conv2D");
                layer_infos.append("Filters: " + str(layer.get_config()["filters"]))
            elif (type(layer) == keras.layers.convolutional.Conv1D):
                layer_types.append("Conv1D");
                layer_infos.append(str(layer.get_config()["filters"]) + " filters")
            elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                layer_types.append("MaxPooling2D");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Dropout):
                layer_types.append("Dropout");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Flatten):
                layer_types.append("Flatten");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Activation):
                #layer_types.append("Activation");
                layer_types.append(layer.get_config()["activation"])
                layer_infos.append("")
            elif (type(layer) == keras.layers.normalization.BatchNormalization):
                layer_types.append("BatchNorm");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Dense):
                layer_types.append("Dense");
                layer_infos.append("Units: "+ str(layer.get_config()["units"]))
            elif (type(layer) == keras.layers.recurrent.LSTM):
                layer_types.append("LSTM");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Reshape):
                layer_types.append("Reshape");
                layer_infos.append("")
            elif (type(layer) == keras.layers.wrappers.TimeDistributed):
                layer_types.append("TimeDistributed");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Lambda):
                layer_types.append("Lambda");
                layer_infos.append("")
            else:
                layer_types.append(layer)
            
        layer_config.append(layer.get_config())
        layer_in_out.append([layer.input, layer.output])

    model_layers.append(layer_types)
    model_layer_infos.append(layer_infos)
    
    # find all unique layer types
    layers_all = []
    for layer in model_layers:
        for x in layer:
            layers_all.append(x)
    
    layers_unique = list(set(layers_all)) 
    print(layers_unique)
    # Choose colors for plot
    selected_colors = get_spaced_colors(len(layers_unique)+4, alpha=0.2) 
      
    gridDX = 30
    gridDY = 30
    offsetDY = 6
    boxheight = 20
    boxwidth = 15
    
    edge = viznet.EdgeBrush('->', lw=5, color='black')
    
    graphs_all = []
    for i in range(num_models):
        brush = viznet.NodeBrush('qc.wide')
        node = brush >> (gridDX*i, gridDX)
        node.text("test", 'bottom')
        
        
        brushes = []
        box0 = viznet.NodeBrush('box', color=None, roundness=0.2, size='large')
        y, ydy = offsetDY+gridDY, offsetDY+gridDY+boxheight
        brushes.append(box0 >> (slice(gridDX*i,gridDX*i+boxwidth), slice(y, ydy)))
        brushes[0].text('Input')
        brushes[0].text('Model no. ' + str(i+1), 'top', fontsize = 18)
        for j in range(len(model_layers[i])):
            color = selected_colors[layers_unique.index(model_layers[i][j])]
            box0 = viznet.NodeBrush('box', color=color, roundness=0.2, size='large')
            y, ydy = offsetDY-gridDY*j, offsetDY-gridDY*j+boxheight
            brushes.append(box0 >> (slice(gridDX*i,gridDX*i+boxwidth), slice(y, ydy)))
            #if j == 0:  # first layer = input layer
             #   brushes[j].text('Input', 'top', fontsize=18)
            brushes[j+1].text(model_layers[i][j])
            brushes[j+1].text(" " + model_layer_infos[i][j], 'right')
            
            edge >> (brushes[j], brushes[j+1])
    
        graphs_all.append(brushes)  

        
    #plt.figure(figsize=(18, 16))
    plt.axis('off')
    #plt.rcParams['figure.figsize'] = [20, 20]
    plt.show()
    
                
        

