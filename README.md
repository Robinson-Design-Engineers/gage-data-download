# gage-data-download
This repository is to document ways to download gage data. The initial notebook published uses NOAA-OWP github's hydrotools package (https://github.com/NOAA-OWP/hydrotools/tree/main) to download both NWS and USGS stream gage data. The NOAA github admin had some helpful advice in this discussion: https://github.com/NOAA-OWP/hydrotools/issues/256. Some code developed by USGS's github is also included - there is some overlap between USGS and NOAA-OWP github functionalities. 

This is not considered an exhaustive notebook and more data retrieval methods are welcomed.

Some possible improvements to explore:
- How to download rainfall data through python packages - it doesn't seem this is easily accessible or clear in NOAA-OWP github. Would need to access NCEI hourly gage data somehow.
- NC FIMAN has a Contrail server for download of state-managed gages - maybe there is someone working in this or NC GDS office with python workflow? Or something we could write? Contrail requires login though. This server has both streamflow and precipitation gages.
- Ambient Weather Network (https://ambientweather.net/) has 5-min interval data for home-owned proprietary weather gages from users. Data can be downloaded in convenient .csv format for a month's data at a time. Historic data older than a year requires a subscription to Ambient Weather Network. Checking this data to nearby NC FIMAN precipitation gages on the Contrail server shows relative agreement. Batch download of all data for one or multiple gages was not explored, but perhaps it could be scraped from the HTML source.
