# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:55:09 2021

@author: jon-f
"""

import numpy as np

def divider(coordinates):

    assert np.array(coordinates).shape[1] == 2, "Coordinates of wrong size [error]"
    def checkifrect(nparray):
        p1, p2, p3, p4, p5 = nparray
        v1_mag = np.linalg.norm(np.subtract(p3, p1))
        v2_mag = np.linalg.norm(np.subtract(p4, p2))
        if np.abs(v1_mag-v2_mag) < 0.001:
            return True
        else:
            return False

    assert checkifrect(coordinates), "The input geometry must be rectangular"
    
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    # plt.plot(x, y, 'bo')
    (c_x, c_y) = (np.sum(x[:-1])/np.size(x[:-1]), np.sum(y[:-1])/np.size(y[:-1]))
    new_polygons = []
    corners = len(coordinates)-1
    
    for i in range(corners):
        polygon = [[x[i], y[i]],
                   [(x[i%corners]+x[(i+1)%corners])/2, (y[i%corners]+y[(i+1)%corners])/2],
                   [c_x, c_y],
                   [(x[i]+x[(i-1)%corners])/2, (y[i]+y[(i-1)%corners])/2],
                   [x[i], y[i]]]
        new_polygons.append(polygon)
    
    # new_polygons = np.array(new_polygons)
    
    return new_polygons

def LLarea(coordinates):
    assert np.array(coordinates).shape[1] == 2, "Coordinates of wrong size [error]"
    p1, p2, p3, p4, p5 = coordinates
    area = np.abs(np.linalg.norm(p2-p1)*np.linalg.norm(p4-p1))
    return area

def allGood(listOfCoords):
    flag = True
    for coords in listOfCoords:
        if LLarea(coords) > 0.1: # arbitary value
            flag = False
            break
    return flag

def reduce(listOfCoords):
    """
    Divides a rectangles defined by closed coordinates into smaller, rectangles.
    Args:
        listOfCoords (list): A list of coordinates in the form [[[x1, y1], [x2, y2], ...., [x1, y1]], ...]. The 
        coordinates must define a rectangular shape.
    Returns:
        new_polygons (list): A set of new rectangular in the form [coordinates1, coordinates2, ..., coordinatesn]
        where n is the number number of length of coordinates-1 (the number of corners)
    
    """
    try:
        listOfCoords[1]
    except:
        def primer(area):
            l1 = [area,]
            l2 = np.array(l1)
            l3 = np.reshape(l2,(l2.shape[0], l2.shape[2], 2))
            return l3
        listOfCoords = primer(listOfCoords)
    assert listOfCoords.shape[2] == 2, "wrong size error"

    if allGood(listOfCoords):
        return listOfCoords
    else:
        newlistOfCoords = []
        for coords in listOfCoords:
            newlistOfCoords = newlistOfCoords + divider(coords)
        
        newlistOfCoords = np.squeeze(np.array(newlistOfCoords))
        return reduce(listOfCoords=newlistOfCoords)


def geoJSONer(coords):
    coords = coords.tolist()
    geoJSON = {
    "type": "FeatureCollection",
    "features": [
        {
          "type": "Feature",
          "properties": {},
          "geometry": {
            "type": "Polygon",
            "coordinates": coords
                      }
        }
      ]
    }
    return geoJSON

def degeoJSONer(geoJSON):
    coords = geoJSON['features'][0]['geometry']['coordinates']
    return coords