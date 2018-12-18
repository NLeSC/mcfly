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
import bokeh.plotting as bk_plot
import bokeh.models as bk_models
import numpy as np
import keras


def model_overview(
        models, 
        max_num_models=6, 
        figure_scale = 0.7,
        file_figure = None,
        extra_layer_info=True):
    """ 
    Visual comparison of the generated deep learning models. 
    The first 'max_num_models' will be displayed.  
    
    Parameters
    ----------
    max_num_models: int
        maximum number of models that should be displayed in one plot
    figure_scale: float
        scales total figure size
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
    fig_width = int(figure_scale * max_num_models * 130)
    num_models = min(len(models), max_num_models)  # number of models to compare 
    maxlength_models = np.max([len(x) for x in model_layers])

    # Choose colors for plot
    RGBA_colors, hex_colors = get_spaced_colors(len(layers_unique)+4) 

    size_x = 1
    box_scaling_x = 0.75
    box_scaling_y = 0.55
    dx = size_x/(num_models) 
    dx_box = dx * box_scaling_x


    dy = 0.4 * dx
    size_y = maxlength_models * dy
    dy_box = dy * box_scaling_y
    dy_arrow = dy * (1-box_scaling_y)
    dy_box_info = 1.5 * dy_box
    dy_header = 1.25*dy

    font_scaling = 0.01*fig_width/max_num_models
    fontsize = str(font_scaling*0.7)+'em'
    header_fontsize = str(font_scaling)+'em'
#    info_fontsize = str(font_scaling*0.55)+'em'

    # Save figure if filanem (file_figure) is given
    if file_figure is not None:
        print("Save model comparison figure to file: '" + file_figure + ".html'")
        bk_plot.output_file(file_figure, title="model_overview figure")

    p = bk_plot.figure(plot_width=fig_width, plot_height=int(size_y/size_x * fig_width))

    # Plot all selected models:
    for i in range(num_models):
        pos_x = np.zeros(len(model_layers[i]))
        pos_y = np.zeros(len(model_layers[i]))
        pos_y_box = np.zeros(len(model_layers[i]))
        widths = []
        heights = []
        box_text = []
        box_text_info = []
        box_color = []

        # Create header 
        header_txt = 'Model ' + str(i+1)
        p.text(x=[dx*i], y=[0], text=[header_txt], text_align='center', text_baseline='middle',
               text_font_size=fontsize)
        subheader_txt = model_types[i]
        p.text(x=[dx*i], y=[-dy/2], text=[subheader_txt] , text_align='center', text_baseline='middle',
               text_font_size=header_fontsize)

        for j in range(len(model_layers[i])):
            widths.append(dx_box)
            # Create small box
            if model_layer_infos[i][j] == "" or (extra_layer_info==False):
                heights.append(dy_box) 
                box_text.append(model_layers[i][j])
                box_text_info.append("")
                dy_box_used = dy_box           
            else: # Create larger box (incl. extra information)
                heights.append(dy_box_info) 
                box_text.append(model_layers[i][j] + "\n " + model_layer_infos[i][j])
                box_text_info.append(model_layer_infos[i][j])
                dy_box_used = dy_box_info

            pos_x[j] = dx*i 
            if j == 0:
                pos_y[j] = -dy_header
            else:
                pos_y[j] = pos_y[j-1]-(dy_box_used + dy_arrow) 

            pos_y_box[j] = pos_y[j] + dy_box_used/2 

            box_color.append('#'+hex_colors[layers_unique.index(model_layers[i][j])])

            # Plot arrows
            p.add_layout(bk_models.Arrow(end=bk_models.NormalHead(fill_color="black", size=10, line_width=0.1),
                               x_start=dx*i, 
                               y_start=pos_y[j], 
                               x_end=dx*i, 
                               y_end=pos_y[j] - dy_arrow))    

        # Actual plotting of boxes and text
        p.rect(x=pos_x, y=pos_y_box, width=widths, height=heights, 
               fill_color=box_color, fill_alpha=0.3)
        p.text(x=pos_x, y=pos_y_box, text=box_text, 
               text_align='center', text_baseline='middle',
               text_font_size=fontsize)

    p.axis.visible = False
    p.grid.visible = False

    bk_plot.show(p)
    




# --------------------------------
# Visualization helper functions:
# --------------------------------

def get_spaced_colors(n, alpha=1):
    """ 
    Helper function. 
    Generate a list of different colors (as RGBA with given alpha, and as hex format)

    """

    max_value = 16581375 #255**3
    interval = int(max_value / n)
    hex_colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    RGBA_colors = [(int(i[1::3], 16)/255, int(i[0::3], 16)/255, int(i[2::3], 16)/255, alpha) for i in hex_colors]    

    return RGBA_colors, hex_colors


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
                layer_types.append("TimeDist");
                layer_infos.append("")
            elif (type(layer) == keras.layers.core.Lambda):
                layer_types.append("Lambda");
                layer_infos.append("")
            else:
                layer_types.append(layer)

        model_layers.append(layer_types)
        model_layer_infos.append(layer_infos)

    return model_types, model_layers, model_layer_infos
