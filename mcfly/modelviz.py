#
# mcfly
#
# Copyright 2017 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
 Summary:
 This module allows to generate a simplified graphical representation
 of the generated Keras/mcfly models. 
 The main callable function is model_overview with 'models' a list of
 Keras-type models as main input.

"""

import matplotlib.pyplot as plt
import numpy as np
import keras

 
def model_overview(
        models, 
        max_num_models=6, 
        scale_figwidth=6,
        scale_boxes=1.5, 
        file_figure = None,
        extra_layer_info=True):
    """ 
    Visual comparison of different deep learning models
    
    Parameters
    ----------
    max_num_models: int
        maximum number of models that should be displayed in one plot
    scale_figwidth: float
        scales total figure size
    scale_boxes: float
        scales size of text and of boxes that represent the layers
    file_figure: str, optional
        if filename is given figure will be saved under that name
        (e.g. "model_comparison.png", or "model_comparison.jpg")
    extra_layer_info: boolean
        if True-> show additional layer information (such as no. of filters)
    """

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
    size_x = scale_figwidth * num_models/5
    scale_y = scale_figwidth
    box_size = scale_boxes*scale_y
    dx = size_x/(num_models + 1)
    dy = 0.06 * scale_y
    dy_infobox = 1.5 * dy
    dy_header = scale_y * 0.15
    arrow_width = 0.002*scale_figwidth

    # set figure size
    if extra_layer_info:
        expected_size_y = dy_header + (1.25*maxlength_models)*dy  
        print("Create figure with extra information")
    else:
        expected_size_y = dy_header + (maxlength_models)*dy 

    fig = plt.figure(figsize=(size_x, max(expected_size_y, scale_y)))

    # Choose colors for plot
    selected_colors = get_spaced_colors(len(layers_unique)+4, alpha=0.2) 

    # Plot layers of all models
    fig_heights = []
    for i in range(num_models):
        pos_x = dx/2 + i * dx
        pos_y = - 0.05*scale_y
        draw_model_header(pos_x, pos_y, expected_size_y, i+1, model_types[i], box_size)

        #initialize pos_y
        pos_y = -dy_header + dy
        for j in range(len(model_layers[i])):
            color = selected_colors[layers_unique.index(model_layers[i][j])]
            
            # check if extra information present
            if model_layer_infos[i][j] == "" or (extra_layer_info==False):
                pos_y = pos_y - dy
                draw_box(pos_x, pos_y, box_size, color, model_layers[i][j])
                if j > 0:
                    draw_arrow(pos_x, pos_y+0.8*dy, 0, dy, arrow_width)
            else:
                pos_y = pos_y - dy_infobox
                draw_box(pos_x, pos_y, box_size, color, model_layers[i][j], text1=model_layer_infos[i][j])
                if j > 0:
                    draw_arrow(pos_x, pos_y+0.8*dy_infobox, 0, dy, 0.01*scale_figwidth)

        fig_heights.append(pos_y)

    # set xlim and ylim, show figure
    plt.axis('equal')
    plt.xlim(-dx, size_x)
    plt.ylim(np.min(fig_heights)-dy, dy)
    plt.axis('off')
    plt.savefig("test.png")
    plt.show()
    # Save figure if filanem (file_figure) is given
    if file_figure is not None:
        print("Save model comparison figure to file: '" + file_figure + "'")
        plt.savefig(file_figure, dpi= 600, bbox_inches='tight')




# --------------------------------
# Visualization helper functions:
# --------------------------------

def draw_box(pos_x, pos_y, size, face_color, text0, text1=None):
    """ 
    Helper function.
    Draw textbox using matplotlib.
    
    Parameters
    ----------
    pos_x: float
        x position of box in plot
    pos_y: float
        y position of box in plot
    size: float
        value to determine size of text and textbox
    face_color: tuple, str
        color value for textbox
    text0: str
        Text to print with textbox
    text1: str, optional
        Additional info-text to print as 2nd line within textbox (default is None)
    """
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
    """ 
    Helper function to draw arrow in plot.
    """

    plt.arrow(pos_x, pos_y, -0.2*dx, -0.2*dy, width=width)


def draw_model_header(pos_x, pos_y, scale_y, model_no, model_type, size):
    """ 
    Helper function. Plot header using strings model number and model_type.

    """

    header_dy = size/500 * scale_y
    plt.text(pos_x, pos_y, "Model "+str(model_no), size=1.5*size, rotation=0,
             horizontalalignment="center", verticalalignment="center")
    plt.text(pos_x, pos_y - header_dy, "type: "+model_type, size=size, rotation=0,
             horizontalalignment="center", verticalalignment="center")


def get_spaced_colors(n, alpha):
    """ 
    Helper function. 
    Generate a list of different colors (as RGBA format) with given alpha

    """

    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    RGBA_colors = [(int(i[1::3], 16)/255, int(i[0::3], 16)/255, int(i[2::3], 16)/255, alpha) for i in colors]    

    return RGBA_colors


def collect_layer_infos(models):
    """ 
    Helper function. 
    Collect infos of model layers (relevant for plotting).

    """
    
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
