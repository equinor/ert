import os.path
import json

from res.enkf.data import ExtParam
from res.enkf.config import ExtParamConfig
from ecl.test import TestAreaContext
from tests import ResTest


class ExtParamTest(ResTest):

    def test_config(self):
        input_keys = ["key1","key2","key3"]
        config = ExtParamConfig("Key" , input_keys)
        self.assertTrue( len(config), 3 )

        for i in range(len(config)):
            self.assertEqual( config[i] , input_keys[i] )

        with self.assertRaises(IndexError):
            c = config[100]

        keys = []
        for key in config.keys():
            keys.append( key )
        self.assertEqual( keys , ["key1","key2","key3"] )

        self.assertIn( "key1" , config )



    def test_data(self):
        input_keys = ["key1","key2","key3"]
        config = ExtParamConfig("Key" , input_keys)
        data = ExtParam( config )

        with self.assertRaises(IndexError):
            d = data[100]

        with self.assertRaises(KeyError):
            d = data["NoSuchKey"]

        self.assertIn( "key1" , data )

        data[0] = 177
        self.assertEqual( data["key1"] , 177 )


        data["key2"] = 321
        self.assertEqual( data[1] , 321 )

        with self.assertRaises(ValueError):
            data.set_vector( [1,2] )

        data.set_vector( [1,2,3] )
        for i in range(len(data)):
            self.assertEqual( i + 1 , data[i] )

        with TestAreaContext("json"):
            data.export( "file.json" )
            d = json.load( open("file.json"))

        for key in data.keys():
            self.assertEqual( data[key] , d[key] )
