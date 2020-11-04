# ET091_06_Solar

## Codes, Files, Folders

### CODiS Data Preprocessing 
CODiS raw data is the opendata from Taiwan meterological observatories. 

- ./StaCrawler.ipynb
    - Prepare the input query for the request
        - Temporary archive: ./WeatherStation.csv
    - Crawl the data from the CODiS website, and categorize by time
        - Temporary archive: ./CODiS
    - Impute the data through all-time interpolation station by station, and all-station KNN time by time
        - Temporary archive: ./fille\_CODiS
    - Scale the data by MinMaxScaler in a range of 0.0 to 1.0
        - Temporary archive: ./scale\_max.csv
        - Temporary archive: ./scale\_min.csv

### Taiwan Map Preprocessing
Taiwan's geometric raw data is an opendata from the Ministry of the Interior. 

- ./GeoGrid.ipynb
    - Load the raw data 
        - Load from ./MapData/TOWN/TOWN\_MOI\_1090820.shp
    - Clip the data by its coordination and the grid mask
        - Temporary archive: ./MapData/taiwan\_grid.shp
        the grid on the Taiwan territory
        - Temporary archive: ./MapData/taiwan\_offgrid.shp
        the grid lying outside the Taiwan territory

### SOLPOS Algorithm Data Preprocessing
SOLPOS raw data is the output from the online calculator.

- ./SolCrawler.ipynb
    - Prepare the input query for the request
    - Crawl the data after the query
- ./SolCrawler.py
    - Function called in the ./FeatMap\_gen.ipynb
    
### FeatMaps Generation
Feature maps are the numpy array of features to be processed by the model

- ./FeatMap\_gen.ipynb
    - Load tempory archives
        - Load from ./fill\_CODiS/\*.csv
        - Load from ./scale\_max.csv
        - Load from ./scale\_min.csv
        - Load from ./MapData/taiwan\_grid.shp
        - Load ./MapData/taiwan\_offgrid.shp
    - Impute the grid's data through all-grid KNN time by time
    - Plot the featmaps for each time slot
    - Clip the axises to the exact scopes
    - Save as .npy file and place in each feature's folders respectively
        - Temporary archive: ./FeatMap/\*/\*.npy
- ./FeatMap\_codis.ipynb
    - Demo for the generation of CODiS's feature maps
- ./FeatMap\_solpos.ipynb
    - Demo for the generation of SOLPOS's feature maps
    
### TruthMap Generation
Truth maps are the numpy array of labels to be processed by the model

- ./TruthMap\_Gen.ipynb
    - Load tempory archives
        - Load from ./fill\_CODiS/\*.csv
        - Load from ./MapData/taiwan\_grid.shp
        - Load ./MapData/taiwan\_offgrid.shp
    - Locate the grids of meteorlogical observatories based on their location in Taiwan_grid
    - Generate the numpy array mask for each grid
        - Temporary archive: ./Grid\_BoolMask/\*.npy
        - Temporary archive: ./TruthMap/BoolMask.npy
    - Generate the TruthMaps by fill in the grid values by the boolean masks
        - Temporary archive: ./TruthMap/GloblRad/\*.npy
        - Temporary archive: ./TruthMap/SunShine/\*.npy

### Satellite Image Preprocessing
Satellite raw images are downloaded from the Meterologix website (Crwaler source code: https://github.com/1010code/python-selenium)
    - Temporary archive: ./satellite-hd-10min

- ./satel_detect.ipynb
    - Check if there are missing images in the folder
    - If so, manually download the images from the website
    
- ./CloudCrop.ipynb
    - Load raw images
        - Load from ./satellite-hd-10min
    - Crop the image array to 200x155
        - Temporary archive: ./CropCloud/\*.png
        
### Model
the model of this project

- ./Model_0.ipynb
    - Build the model
    - Define custom loss functions
        - Load from ./TruthMap/BoolMask.npy
    - Input data and reshape into the exact dimensions (24x3xchannelx200x155)
        - Load from ./FeatMap/\*/\*.npy
        - Load from ./CropCloud/\*.png
        - Load from ./TruthMap/GloblRad/\*.npy
        - Load from ./TruthMap/SunShine/\*.npy
    - Train and predict
        - Temporary archive: ./Model_mcp
        - Temporary archive: ./Tensorboard
- ./Model.py
    - Function called in the ./Model_0.ipynb

## Collaboration

### ET091027:
    - Initiative and Proposal
    - Compile the initial stage of the "CODiS Data Preprocessing"
    - Compile all the other project's codes besdies the above-mentioned
    - Poster
    - Final report
    - Presentation
    
### ET091011:
    - Data Crawling
    - Data preprocessing
    - Power Point
    - Final report
    - Presentation