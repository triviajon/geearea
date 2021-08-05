# geearea
A Python package for easily managing and downloading Google Earth Engine data defined over geographical area.

## Installation 
To install using the `pip` package manager, simply run:
```pip install geearea```
If not found, pip will also automatically try to install the following Python depencancies:
```
folium
earthengine-api
matplotlib
numpy
```
The only optional package is `GDAL`, which is known to be a problematic install. The only use of GDAL in this package is for downloading singular large ee.Image objects as GeoTIFFs using the `.download` method of the `Area` object. The image is geographically split into small rectangular, and then GDAL automatically mosaicks them back into a singular TIF file. Without it, all images will automatically be downloded to Google Cloud Storage.

## Tutorial

To use this package, start by importing. On import, you must authenticate to Google Earth Engine's servers. You can authenticate by providing a service account and private key json file. Either manually type this in on import, or place a file called "service_account.txt" in the working directory. Otherwise, you will have to manually input the service account email (see below) and place the private key json file in the current working directory for it to be automatically found:
```
In [1]: import geearea.area as ga
Enter your service account email: jon***@***.iam.gserviceaccount.com
```

Define a rectangular area of interest using the public tool https://geojson.io/ and save it as a variable. Then, create the object using the `Area` object:
```
In [2]: geoJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              -63.91845703124999,
              -68.73638345287264
            ],
            [
              -59.08447265624999,
              -68.73638345287264
            ],
            [
              -59.08447265624999,
              -67.08455048507471
            ],
            [
              -63.91845703124999,
              -67.08455048507471
            ],
            [
              -63.91845703124999,
              -68.73638345287264
            ]
          ]
        ]
      }
    }
  ]
}

In [3]: larsen = ga.Area(geoJSON)
```
Once the Area object is created, you can use the methods provided to work with it:

```
In [4]: larsen_all_images = larsen.apply_filters(start_date='2021-06-20', end_date='2021-06-28')
In [5]: larsen.download(larsen_all_images, directory='export', scale=50)
```

This package also includes support for custom image assets uploaded to Google Cloud. This requires the setup of a Google Code Editor Folder. This can just be done by going to https://code.earthengine.google.com/ and clicking on the "assets" tab. From there, just add a username and create the base folder. From there, just follow the same general steps to create the object:

```
In [6]: Rasters = ga.CustomCollection('glaciers_ee', 'synthetic_rasters', 'jonfrosario')
In [7]: Rasters_all = Rasters.apply_filters()
In [8]: Rasters_filtered = Rasters.gammafilter(Rasters_all)
```


## Cloud Project and Service Account Setup
To create a Google Cloud Project: 
https://developers.google.com/earth-engine/earthengine_cloud_project_setup#create-a-cloud-project

Once your project is created (or you have been invited to one), follow the instructions here on "Create a service account" and "Create a private key for the service account" to create a service account:
https://developers.google.com/earth-engine/guides/service_account

Finally, just add your `privatekey.json` file to the working directory where you are using the `geearea` package. Note: the private key does not need to be named `privatekey.json`. 

## Known Issues
- The package is not modular
  - I was having build issues when the package was modular, so I decided to compile all of the src code into `area.py`. If you manage to build successfully and test the code in a modular format, please reach out.
- Regular tiffs cause issues with Earth Engine causing them not to be downloadable

