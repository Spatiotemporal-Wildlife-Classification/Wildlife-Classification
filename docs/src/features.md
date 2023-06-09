# Features
This module aims to collect, format, and extract metadata features from the original observations. 
For a comprehensive breakdown of the collected features, please review the Spatiotemporal breakdown 
within the [Dataset](../dataset.md).

Please note, the below files work in collaboration with the Distributed Scraping Network (DSN) repository, 
and the DSN-leaf repository. The combination of these repositories allows for the collection of weather data from Open-Meteo 
using two remove machines: 

1. Main server providing an API specifying the observations to collect weather data for, and storing all collected information
2. A leaf node, allowing for the collection of weather data from Open-Meteo and transferring it to the mains server.

The combination of these repositories allows for the collection of data on a remote device, enabling collection on any machine in a simple, easy manner. 
Please respect the rate limits of Open-Meteo at 10000 requests per day.

Please review the documentation for the above two repositories within for more details.

Feature contains two files, please select file to review the documentation:

1. [Weather Scraping Node](../src/features/dsn_leaf_node.md)
2. [Weather Collection](../src/features/dsn_weather_collection.md)
