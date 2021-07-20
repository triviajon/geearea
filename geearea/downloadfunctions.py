# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:56:08 2021

@author: jon-f
"""

import os
import glob
import urllib
import ee.batch
import time
import reducer

# downlaod functions
def DLimg(image, directory, scale, aoi):
    """
    Downloads an image or image collection.
    
    Args:
        images (ee.Image or ee.ImageCollection): images to download
        directory (str): name of relative directory to save images to, 
        leave as a blank string you'd like to save in current directory

    Returns:
        None.

    """
    
    imageID = image.id().getInfo()
    def nextfile(imageID):
        def filenamer(imageID, i):
            name = imageID + '_' + str(i) + ".tif"
            filename = os.path.join(os.getcwd(), directory, name)
            return filename
        
        i = 0
        while os.path.isfile(filenamer(imageID, i)):
            i += 1
        return filenamer(imageID, i), i
    
    url = image.clip(aoi).getThumbURL({'format': 'geotiff', 'scale': scale})

    filepath, i_stop = nextfile(imageID)
    urllib.request.urlretrieve(url, filepath)
    print(url, "has been downloaded to", str(filepath))

def merger(directory='export'):
    """
    Run this function after the download is complete to have gdal attempt to merge the geoTIFFs. 
    Generally should not used on it's own.
    
    Args:
        directory (str): The directory to find the unmerged TIF images in. Defaults to 'export'
        
    Returns:
        None.
    
    """
    tiles = glob.glob((directory+'/*.tif'))
    tiles = " ".join(tiles)
    
    name = directory + "/" + tiles[tiles.find('/')+1:tiles.find('.tif')-2] + ".tif"
    print(name)
    
    os.system('!gdal_merge.py -o $name $tiles')
    
    if os.path.isfile(name):
        for file in os.scandir(directory):
            if file.path.endswith('.tif') and file.path.count('_')==9:
                print(file.path, 'has been deleted!')
                os.remove(file.path)
    else:
        assert True, "gdal_merge was not found, try restarting the kernel and running merger(directory)"
        
def batchExport(images, scale, directory='test', tryReduce=False):
    """
    Creates a number of ee.batch tasks equal to the number of ee.Image objects in images.
    The images are downloaded to the Google Cloud Platform Storage glaciers_ee bin in the subdirectory set.

    Args:
        images (list): a list of images to export to cloudstorage
        scale (int): the scale in meters/pixel to download the images at
        directory (str): the subdirectory to download the images to in the glaciers_ee bin. Defaults to 'export'
        
    Returns:
        None.
    """
    tasks = []
    active_ind = []
    n = 0
    for img in images:
        name = img.get('system:index').getInfo()
        task = ee.batch.Export.image.toCloudStorage(**{
            'image': img,
            'fileNamePrefix': directory + '/' + name,
            'scale': scale,
            'crs': 'EPSG:3031',
            'bucket': 'glaciers_ee',
            'fileFormat': 'GeoTIFF',
            'maxPixels': 10e12
        })
        print(name)
        task.start()
        tasks.append(task)
        active_ind.append(task.active())
    
    while True in active_ind:
        rate = 3230 # mb/hr 
        num_imgs = len(active_ind)    
        time_for_completion = 180/rate * 30/scale * num_imgs
        units = 'hours'

        if time_for_completion < 1.0:
            time_for_completion *= 60
            units = 'minutes'

        print('The approximate completion time is: {:.2f} {}.'.format(time_for_completion, units))
        new_tasks = []
        for i, task in enumerate(tasks):
#             try:
            print('{} is {}'.format('Task #' + task.id, task.status()['state']).lower())
            if task.status()['state'].lower() == "failed":
                image = images[i]
                image_segmented = reducer.reduce(reducer.primer(image.get('system:footprint').getInfo()['coordinates']))
                tasks.pop(i)
                active_ind.pop(i)
                print('{} has failed.'.format(task.id))
                print('Adding {} new segemented images to be exported.'.format(len(image_segmented.tolist())))
                for coords in image_segmented.tolist():
                    name_n = name + '_' + str(n)
                    task = ee.batch.Export.image.toCloudStorage(**{
                        'image': image,
                        'region': ee.Geometry.Polygon([coords]),
                        'fileNamePrefix': directory + '/' + name_n,
                        'scale': scale,
                        'crs': 'EPSG:3031',
                        'bucket': 'glaciers_ee',
                        'fileFormat': 'GeoTIFF',
                        'maxPixels': 10e12
                    })
                    images.append(image)
                    task.start()
                    new_tasks.append(task)
                    active_ind.append(task.active())
                    n += 1
#             except:
#                 print('error!')
            active_ind[i] = task.active()
        tasks = tasks + new_tasks
        print('----------------------------------------------------')
        time.sleep(5*active_ind.count(True)**(1/2))