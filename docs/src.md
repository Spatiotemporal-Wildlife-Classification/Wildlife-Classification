### Code Structure
The code is structured within 5 directories:
1. [Data](src/data.md)
2. Ensemble
3. Features
4. Models
5. Structure

#### Data
The data directory serves to process, clean, and structure the data sourced from [iNaturalist](https://www.inaturalist.org/). 
The cleaning process forms a Data Pipeline, writing the processed data into 
the `data/interim` directory within the project root folder, for further feature extraction.

