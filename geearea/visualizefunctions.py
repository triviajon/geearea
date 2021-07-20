# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:57:34 2021

@author: jon-f
"""
import folium
import ee
import matplotlib.pyplot as plt
import numpy as np


# extra functions
# folium maps
def add_ee_layer(self, ee_image_object, vis_params, name):
    """
    Adds an ee.Image layer to a folium map
    
    Args: 
        ee_image_object (ee.Image): An image to place on folium map
        vis_params (dict): Visual parameters to display the image. See GEE "Image Visualization"
        name (str): Name of layer on folium map
        
    Returns:
        None.
        
    """
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles = map_id_dict['tile_fetcher'].url_format,
        attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name = name,
        overlay = True,
        control = True
    ).add_to(self)
  
folium.Map.add_ee_layer = add_ee_layer

        
def eeICtolist(imageCollection):
    """
    Converts an ee.ImageCollection to a python list of ee.Image objects. 
    
    Args:
        imageCollection (ee.ImageCollection): the collection to be converted to a list

    Returns:
        images (list): the list of images in the ee.ImageCollection
        
    """
    size = imageCollection.size()
    eeList = imageCollection.toList(size)
    aslist = []
    for i in range(size.getInfo()):
        aslist.append(ee.Image(eeList.get(i)))
        
    return aslist

def dispHist(hists):
    """
    Plots 1 histogram for each histogram passed into it. 
    
    Args:
        hists (list): Can either be a histogram from Area.hist() or a list of histograms [Area.hist(img1), Area.hist(img2)]

    Returns:
        None.
    """
    assert type(hists) is list, "Type of hists should be a list from self.hist, or a list of lists"
    a = np.array(hists)
    if a.ndim == 3:
        # case for hists is a list of histograms
        for hist in hists:
            a = np.array(hist)
            x = a[:, 0]
            y = a[:, 1]/np.sum(a[:, 1])
            plt.grid()
            plt.plot(x, y, '.')
            plt.show()
    else:
        # case for hists is single histogram
        x = a[:, 0]
        y = a[:, 1]/np.sum(a[:, 1])
        plt.grid()
        plt.plot(x, y, '.')
        plt.show()
        
def get_dates(images):
    """    
    Returns the dates of each image in imageCllection
    
    Args:
        images (ee.Image or ee.ImageCollection): Image/images to get the date of

    Returns:
        dates (str or list): The dates that the images were taken on
    """
    try:
        if type(images) == ee.ImageCollection:
            dates = ee.List(images
                            .aggregate_array('system:time_start')
                            .map(lambda time_start:
                                 ee.Date(time_start).format('Y-MM-dd'))).getInfo()
        elif type(images) == ee.Image:
            dates = ee.Date(images.get('system:time_start').getInfo()).format("yyyy-MM-dd").getInfo()

        else: 
            assert False, "Image is not of type ee.Image or ee.ImageCollection"
        return dates
    except:
        print('ERROR: Something went wrong and the Image does not have the system:time_start property.')
        print('If you need the date, you can manually retrieve it from the system:index property')
        return None
