import numpy as np
from geopy.geocoders import Nominatim
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam

class coord2loc(Node):
    def config_input_slots():
        return {
            "latitude": DataType.ARRAY,
            "longitude": DataType.ARRAY
        }

    def config_output_slots():
        return {
            "coord_info": DataType.TABLE,
        }

    def setup(self):
        # list keys : ['ISO3166-2-lvl4', 'ISO3166-2-lvl6', 'ISO3166-2-lvl7', 'city',
        # 'city_district', 'country', 'country_code', 'county', 'district', 'hamlet', 'locality',
        # 'municipality', 'postcode', 'region', 'road', 'state', 'state_district', 'suburb', 'town', 'village']
        # Use a user-defined user agent to comply with Nominatim's usage policy
        self.road = ['road']
        self.village = ['village', 'hamlet']

        self.city = ['city', 'town', 'locality', 'municipality', 'city_district']
        self.state = ['state', 'region', 'state_district']
        self.country = ['country']
        self.geolocator = Nominatim(user_agent='my_application')

    def process(self, latitude: Data, longitude: Data):
        if latitude is None or latitude.data is None or longitude is None or longitude.data is None:
            return None
        # Check if both latitude and longitude data are provided
        latitude = latitude.data
        longitude = longitude.data

        # Convert the input coordinates to a string format "latitude, longitude"
        coordinates = (latitude, longitude)        
        # Use geolocator to find the location from coordinates
        location = self.geolocator.reverse(coordinates)
        
        # Check if location is found
        if location and location.address:

            location_info = {}
            for key in self.city:
                value = location.raw.get('address', {}).get(key)
                if value:
                    location_info['city'] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.state:
                value = location.raw.get('address', {}).get(key)
                if value:
                    location_info['state'] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.country:
                value = location.raw.get('address', {}).get(key)
                if value:
                    location_info['country'] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.road:
                value = location.raw.get('address', {}).get(key)
                if value:
                    location_info['road'] = Data(DataType.STRING, value, {})
                    break
            value = None
            for key in self.village:
                value = location.raw.get('address', {}).get(key)
                if value:
                    location_info['village'] = Data(DataType.STRING, value, {})
                    break
            location_info['full_address'] = Data(DataType.STRING, location.address, {})
            value = None

            return {"coord_info": (location_info, {})}
        else:
            return {"coord_info": ({'info':Data(DataType.STRING, "No address found", {})}, {})}

