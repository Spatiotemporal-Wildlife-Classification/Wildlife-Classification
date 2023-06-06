# Dataset

The dataset encompasses the observations and the encompassing spatiotemporal snapshot. 
Each section below details the relevant essential information regarding the components of the dataset, and the 
final dataset taxonomic count breakdown for observations and images. Note the distinction between observations and images
as observations may contain erroneous or mislabelled images. 

## Observations

## Spatiotemporal 
The spatiotemporal data is sourced from [Open-Meteo Weather API](https://open-meteo.com/). 
The table below details the set of collected spatiotemporal metdata per observation in order to generate the 
spatiotemporal snapshot. 

For more information on the Open-Meteo Historic API please visit
[Open-Meteo Historic Weather API](https://open-meteo.com/en/docs/historical-weather-api). The table below uses information 
from the API documentation to describe some collected features. 
Additionally, the table details additional information extracted from the 
metadata such as day/ night, light/ dark, terrestrial, etc. 

| Feature                       | Description                                                                                                             | Unit/ Format                              | Timeframe |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------|
| Observed on                   | Time of observation                                                                                                     | ISO8601                                   | Instant   |
| Coordinates                   | World Geodetic System (WGS84)                                                                                           | (latitude, longitude)                     | Instant   |
| Positional accuracy           | Publicly available positional accuracy                                                                                  | $R$                                       | Instant   |
| Elevation                     | Meters above sea level                                                                                                  | Meters (m)                                | Instant   |
| Ground temperature (2m)       | Air temperature 2 meters above ground                                                                                   | Celsius ($^{\circ}$ C)                    | Hourly    |
| Relative humidity (2m)        | Humidity 2 meters above ground                                                                                          | Percentage (%)                            | Hourly    |
| Dew point (2m)                | Dew point 2 meters above ground                                                                                         | Celsius ($^{\circ}$ C)                    | Hourly    |
| Apparent temperature          | Real feel temperature considering additional factors                                                                    | Celsius ($^{\circ}$ C)                    | Hourly    |
| Surface pressure              | Atmospheric air pressure at the surface.                                                                                | Hectopascal (hPa)                         | Hourly    |
| Cloudcover                    | Cloudcover within the immediate area                                                                                    | % of area covered                         | Hourly    |
| Low cloudcover                | Cloudcover and fog up to an altitude of $3$ kilometers                                                                  | % of area covered                         | Hourly    |
| Mid cloudcover                | Cloudcover from $3-8$ kilometers altitude                                                                               | % of area covered                         | Hourly    |
| High cloudcover               | Cloudcover from $8$ kilometers altitude                                                                                 | % of area covered                         | Hourly    |
| Wind speed (10m)              | Wind speed at 10 meters above ground                                                                                    | kilometers per hour (km/h)                | Hourly    |
| Wind speed (100m)             | Wind speed at 100 meters above ground                                                                                   | kilometers per hour (km/h)                | Hourly    |
| Wind direction (10m)          | Wind direction at 10 meters above ground                                                                                | Degrees (${\circ}$)                       | Hourly    |
| Wind direction (100m)         | Wind direction at 100 meters above ground                                                                               | Degrees ($\circ$)                         | Hourly    |
| Wind gusts (10m)              | Maximum wind speed of the preceding hour                                                                                | kilometers per hour (km/h)                | Hourly    |
| Shortwave radiation           | Average shortwave solar radiation of the preceding hour                                                                 | Watt per square meter (W/m$^2$)           | Hourly    |
| Direct radiation              | Average direct solar radiation of the preceding hour                                                                    | Watt per square meter (W/m$^2$)           | Hourly    |
| Diffuse radiation             | Average diffuse solar radiation of the preceding hour                                                                   | Watt per square meter (W/m$^2$)           | Hourly    |
| Vapor pressure dificit        | A high VPD effects the transpiration of plants                                                                          | Kilopascal (kPa)                          | Hourly    |
| Evapotranspiration            | Water evaporation into the atmosphere                                                                                   | Millimeters (mm)                          | Hourly    |
| ET0 FAO Evapotranspiration    | Metric estimating required irregation for plants                                                                        | Millimeters (mm)                          | Hourly    |
| Precipitation                 | Hourly precipitation sum (rain, showers, snow)                                                                          | Millimeters (mm)                          | Hourly    |
| Snowfall                      | Hourly snowfall sum                                                                                                     | Centimeters (cm)                          | Hourly    |
| Rain                          | Large scale weather systems resulting rain                                                                              | Millimeters (mm)                          | Hourly    |
| Hourly Weather code           | [WMO](https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM) numeric weather code | WMO code                                  | Hourly    |
| Soil temperature (0cm-7cm)    | Temperature in the soil at 0-7 centimeters                                                                              | Celsius ($^{\circ}$)                      | Hourly    |
| Soil temperature (7cm-28cm)   | Temperature in the soil at 7-28 centimeters                                                                             | Celsius ($^{\circ}$)                      | Hourly    |
| Soil temperature (28cm-100cm) | Temperature in the soil at 28-100 centimeters                                                                           | Celsius ($^{\circ}$)                      | Hourly    |
| Soil moisture (0cm-7cm)       | Average water content in the soil at 0-7 centimeters                                                                    | Meter cubed per meter cubed (m$^3$/m$^3$) | Hourly    |
| Soil moisture (7cm-28cm)      | Average water content in the soil at 7-28 centimeters                                                                   | Meter cubed per meter cubed (m$^3$/m$^3$) | Hourly    |
| Soil moisture (28cm-100cm)    | Average water content in the soil at 28-100 centimeters                                                                 | Meter cubed per meter cubed (m$^3$/m$^3$) | Hourly    |
| Daily Weather code            | [WMO](https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM) numeric weather code | WMO code                                  | Daily     |
| Max temperature (2m)          | Maximum daily temperature at 2 meters above ground                                                                      | Degrees ($\circ$)                         | Daily     |
| Min temperature (2m)          | Minimum daily temperature at 2 meters above ground                                                                      | Degrees ($\circ$)                         | Daily     |
| Apparent temperature max      | Maximum real-feel temperature at 2 meters above ground                                                                  | Degrees ($\circ$)                         | Daily     |
| Apparent temperature min      | Minimum real-feel temperature at 2 meters above ground                                                                  | Degrees ($\circ$)                         | Daily     |
| Precipitation sum             | The sum of daily precipitation (rain, showers, snowfall)                                                                | Millimeters (mm)                          | Daily     |
| Rain sum                      | Sum of daily rain                                                                                                       | Millimeters (mm)                          | Daily     |
| Snowfall sum                  | Sum of daily snowfall                                                                                                   | Centimeters (cm)                          | Daily     |
| Precipitation hours           | The number of hours with rain in a day                                                                                  | $Z$                                       | Daily     |
| Sunrise                       | Local sunrise time                                                                                                      | ISO 8601                                  | Daily     |
| Sunset                        | Local sunset time                                                                                                       | ISO 8601                                  | Daily     |
| Wind speed max (10m)          | Maximum daily wind speed 10 meters above ground                                                                         | Kilometers per hour (km/h)                | Daily     |
| Wind gusts (10m)              | Maximum daily gust speed at 10 meters above ground                                                                      | Kilometers per hour                       | Daily     |
| Dominant wind direction       | Dominant daily wind direction for winds at 10 meters                                                                    | Kilometers per hour (km/h)                | Daily     |
| Shortwave radiation sum       | The daily sum of short wave radiation                                                                                   | Megajoules per meter squared (MJ/m$^2$)   | Daily     |
| Daily evapotranspiration      | Sum of daily evapotranspiration                                                                                         | Millimeters (mm)                          | Daily     |
| Terrestrial                   | Terrestrial or aquatic observation                                                                                      | $\{0, 1\}$                                | Instant   |
| Hemisphere                    | Location lies in the northern/ southern hemisphere                                                                      | $\{0, 1\}$                                | Instant   |
| Day                           | Sighting occurrence in light/ dark                                                                                      | $\{0, 1\}$                                | Instant   |
| Season                        | Season of sighting, dependent on hemisphere                                                                             | Season                                    | Instant   |

