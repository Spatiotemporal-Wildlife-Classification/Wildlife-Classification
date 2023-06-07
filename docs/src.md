# Code Structure
The code is structured within 5 directories:

1. [Data](src/data.md)

    A module performing the cleaning, processing, and structuring of the iNaturalist observations

2. Ensemble

    A module detailing the novel cascading ensemble classifier and performing its evaluation against a validation set.

3. [Features](src/features.md)

    A module performing feature extractions from both iNaturalist observations and Open-Meteo weather data.

4. Models 

    A module detailing the construction, training, and evaluation of both image and meta classifiers

5. Structure

    A module enabling access to the projects structure for easy file access.

#### Data
The data directory serves to process, clean, and structure the data sourced from [iNaturalist](https://www.inaturalist.org/). 
The cleaning process forms a Data Pipeline, writing the processed data into 
the `data/interim` directory within the project root folder, for further feature extraction.

