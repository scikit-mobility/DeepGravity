import sys
import numpy as np
import sqlalchemy.types
import pandas as pd
import geopandas as gpd
import fiona
import zipfile
import json
import shapely
import shapely.wkt
import area
from math import sqrt, sin, cos, pi, asin

import statanal.dbutils as db


USER = 'user'

# Paths

osm_dir = '/path/to/osm/dir/'
uk_dir = '/path/to/uk/data/'
sez_dir = '/path/to/uk/data/'


class NumpyEncoder(json.JSONEncoder):
    """
    Enable to save numpy arrays to json files as lists

    Example
    -------

        with open(fname, 'w') as f:
            json.dump(np.array([1,2,3]), f, cls=NumpyEncoder)
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)



def earth_distance(lat_lng1, lat_lng2):
    """
    Compute the distance (in km) along earth between two lat/lon pairs
    :param lat_lng1: tuple
        the first lat/lon pair
    :param lat_lng2: tuple
        the second lat/lon pair

    :return: float
        the distance along earth in km
    """
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...


def getbbox(x):
    lngs_lats = x.exterior.xy
    return min(lngs_lats[0]), min(lngs_lats[1]), max(lngs_lats[0]), max(lngs_lats[1])


def from_tessellation_to_tess_bbox(tessellation, fname):
    tess_bbox = np.array(tuple(tessellation.geometry.apply(getbbox).values))
    df = pd.DataFrame(tess_bbox, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    df['tile_ID'] = tessellation['tile_ID']
    df.to_csv(fname, index=False)


def bbox_to_polygon(bbox):
    min_lat, max_lat, min_lon, max_lon = bbox
    bbox2 = [[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat],\
             [max_lon, min_lat], [min_lon, min_lat] ]
    return bbox2


def poly_area_km2(vertices):
    """
    Example
    -------

    Compute areas of polygons in a geodataframe

        [poly_area_km2(list(zip(*w[0].exterior.xy))) for w in gdf.way]

    """
    return area.area({ "type": "Polygon", "coordinates": [vertices]}) / 1e6


def get_area_km2(ww):
    if type(ww) == shapely.geometry.polygon.Polygon:
        return poly_area_km2(list(zip(*ww.exterior.xy)))
    else:
        return sum([poly_area_km2(list(zip(*w.exterior.xy))) for w in ww])


def line_length_km(vertices):
    """
    Example
    -------

    Compute lengths of linestrings in a geodataframe

        [line_length_km(list(zip(*w.xy))) for w in gdf.way]

    """
    return sum([earth_distance(a, b) for a, b in zip(vertices[1:], vertices[:-1])])


def get_length_km(ww):
    if type(ww) == shapely.geometry.linestring.LineString:
        return line_length_km(list(zip(*ww.xy)))
    else:
        return sum([line_length_km(list(zip(*w.xy))) for w in ww])


# Handmade features
# ------------------

feature2process_funcs = {
    # roads
    'residential_line' : [(lambda x: x.way.apply(get_length_km).sum())],
    'other_road_line' : [(lambda x: x.way.apply(get_length_km).sum())],
    'main_road_line' : [(lambda x: x.way.apply(get_length_km).sum())],
    # land use
    'residential_landuse' : [(lambda x: x.way.apply(get_area_km2).sum())],
    'commercial_landuse' : [(lambda x: x.way.apply(get_area_km2).sum())],
    'industrial_landuse' : [(lambda x: x.way.apply(get_area_km2).sum())],
    'retail_landuse' : [(lambda x: x.way.apply(get_area_km2).sum())],
    'natural_landuse' : [(lambda x: x.way.apply(get_area_km2).sum())],
    # food
    'food_point' : [(lambda x: x.way.count())],
    'food_poly' : [(lambda x: x.way.apply(get_area_km2).sum()), (lambda x: x.way.count())],
    # retail
    'retail_point' : [(lambda x: x.way.count())],
    'retail_poly' : [(lambda x: x.way.apply(get_area_km2).sum()), (lambda x: x.way.count())],
    # schools
    'school_point' : [(lambda x: x.way.count())],
    'school_poly' : [(lambda x: x.way.apply(get_area_km2).sum()), (lambda x: x.way.count())],
    # health
    'health_point' : [(lambda x: x.way.count())],
    'health_poly' : [(lambda x: x.way.apply(get_area_km2).sum()), (lambda x: x.way.count())],
    # transport
    'transport_point' : [(lambda x: x.way.count())],
    'transport_poly' : [(lambda x: x.way.apply(get_area_km2).sum()), (lambda x: x.way.count())]
}

feature2db_table = {
    # roads
    'residential_line' : 'query_line',
    'other_road_line' : 'query_line',
    'main_road_line' : 'query_line',
    # land use
    'residential_landuse' : 'query_poly',
    'commercial_landuse' : 'query_poly',
    'industrial_landuse' : 'query_poly',
    'retail_landuse' : 'query_poly',
    'natural_landuse' : 'query_poly',
    # food
    'food_point' : 'query_point',
    'food_poly' : 'query_poly',
    # retail
    'retail_point' : 'query_point',
    'retail_poly' : 'query_poly',
    # schools
    'school_point' : 'query_point',
    'school_poly' : 'query_poly',
    # health
    'health_point' : 'query_point',
    'health_poly' : 'query_poly',
    # transport
    'transport_point' : 'query_point',
    'transport_poly' : 'query_poly'
}


def db_set(lst, query_str=' IN'):
    return ' ' + query_str + ' ({})'.format(', '.join(["'%s'" % t for t in lst]))


feature2query = {
    # lines
    'residential_line': ['planet_osm_line',
                         {'highway': db_set(['residential'])
                          }],
    'other_road_line': ['planet_osm_line',
                        {'highway': db_set(['primary', 'secondary', 'tertiary',
                                            'unclassified', 'service', 'primary_link',
                                            'secondary_link', 'tertiary_link',
                                            'living_street', 'pedestrian', 'track', 'road'])
                         }],
    'main_road_line': ['planet_osm_line',
                       {'highway': db_set(['motorway', 'trunk',
                                           'motorway_link', 'trunk_link'])
                        }],

    # landuse polygons
    'residential_landuse': ['planet_osm_polygon',
                            {'landuse': db_set(['residential'])
                             }],
    'commercial_landuse': ['planet_osm_polygon',
                           {'landuse': db_set(['commercial'])
                            }],
    'industrial_landuse': ['planet_osm_polygon',
                           {'landuse': db_set(['industrial', 'garages', 'port', 'quarry'])
                            }],
    'retail_landuse': ['planet_osm_polygon',
                       {'landuse': db_set(['retail'])
                        }],
    'natural_landuse': ['planet_osm_polygon',
                        {'landuse': db_set(['farmland', 'farmyard', 'forest', 'grass',
                                            'greenfield', 'greenhouse_horticulture',
                                            'meadow', 'orchard', 'plant_nursery',
                                            'recreation_ground', 'village_green', 'vineyard']),
                         'leisure': db_set(['park', 'garden', 'common', 'dog_park',
                                            'nature_reserve', 'playground']),
                         'boundary': db_set(['national_park', 'protected_area']),
                         'building': db_set(['greenhouse'])
                         }],
    #                          'natural' : ['wood']}],

    # points + poly
    'processing_point': ['planet_osm_point',
                         {'man_made': db_set(['wastewater_plant', 'landfill', 'works'])
                          }],
    'processing_poly': ['planet_osm_polygon',
                        {'man_made': db_set(['wastewater_plant', 'landfill', 'works']),
                         'power': db_set(['plant', 'substation'])
                         }],
    'security_point': ['planet_osm_point',
                       {'amenity': db_set(['fire_station', 'police', 'prison']),
                        'military': db_set(['naval_base', 'barracks', 'office'])
                        }],
    'security_poly': ['planet_osm_polygon',
                      {'amenity': db_set(['fire_station', 'police', 'prison']),
                       'military': db_set(['naval_base', 'barracks', 'office'])
                       }],
    'leisure_point': ['planet_osm_point',
                      {'leisure': db_set(['horse_riding', 'fishing', 'summer_camp',
                                          'fitness_centre', 'ice_rink', 'horse_riding',
                                          'marina', 'pitch', 'sports_centre', 'track',
                                          'swimming_pool', 'water_park', 'beach_resort']),
                       'sport': ' IS NOT NULL '
                       }],
    'leisure_poly': ['planet_osm_polygon',
                     {'leisure': db_set(['horse_riding', 'fishing', 'summer_camp',
                                         'fitness_centre', 'ice_rink', 'horse_riding',
                                         'marina', 'pitch', 'sports_centre', 'track',
                                         'swimming_pool', 'water_park', 'beach_resort']),
                      'sport': ' IS NOT NULL ',
                      'building': db_set(['sports_hall'])
                      }],
    'airport_point': ['planet_osm_point',
                      {'aeroway': db_set(['aerodrome'])
                       }],
    'airport_poly': ['planet_osm_polygon',
                     {'aeroway': db_set(['aerodrome'])
                      }],
    'hotel_point': ['planet_osm_point',
                    {'tourism': db_set(['apartment', 'alpine_hut', 'camp_site',
                                        'caravan_site', 'chalet', 'guest_house',
                                        'hostel', 'hotel', 'motel', 'wilderness_hut'])
                     }],
    'hotel_poly': ['planet_osm_polygon',
                   {'tourism': db_set(['apartment', 'alpine_hut', 'camp_site',
                                       'caravan_site', 'chalet', 'guest_house',
                                       'hostel', 'hotel', 'motel', 'wilderness_hut']),
                    'building': db_set(['hotel'])
                    }],
    'tourist_point': ['planet_osm_point',
                      {'historic': db_set(['archaeological_site', 'church', 'monument']),
                       'tourism': db_set(['aquarium', 'attraction', 'gallery', 'museum',
                                          'theme_park', 'zoo'])
                       }],
    'tourist_poly': ['planet_osm_polygon',
                     {'historic': db_set(['archaeological_site', 'church', 'monument']),
                      'tourism': db_set(['aquarium', 'attraction', 'gallery', 'museum',
                                         'theme_park', 'zoo'])
                      }],
    'religious_point': ['planet_osm_point',
                        {'amenity': db_set(['place_of_worship'])
                         }],
    'religious_poly': ['planet_osm_polygon',
                       {'amenity': db_set(['place_of_worship']),
                        'building': db_set(['religious', 'cathedral', 'church', 'mosque',
                                            'temple', 'synagogue'])
                        }],

    'food_point': ['planet_osm_point',
                   {'amenity': db_set(['bar', 'biergarten', 'cafe', 'fast_food',
                                       'food_court', 'ice_cream', 'pub', 'restaurant']),
                    'shop': db_set(['alcohol', 'bakery', 'beverages',
                                    'brewing_supplies', 'butcher', 'cheese',
                                    'chocolate', 'coffee', 'confectionery',
                                    'convenience', 'deli', 'dairy', 'farm',
                                    'frozen_food', 'greengrocer', 'health_food',
                                    'ice_cream', 'organic', 'pasta', 'pastry',
                                    'seafood', 'spices', 'tea', 'water',
                                    'department_store', 'general', 'kiosk', 'mall',
                                    'supermarket', 'wholesale'])
                    }],
    'food_poly': ['planet_osm_polygon',
                  {'amenity': db_set(['bar', 'biergarten', 'cafe', 'fast_food',
                                      'food_court', 'ice_cream', 'pub', 'restaurant']),
                   'shop': db_set(['alcohol', 'bakery', 'beverages',
                                   'brewing_supplies', 'butcher', 'cheese',
                                   'chocolate', 'coffee', 'confectionery',
                                   'convenience', 'deli', 'dairy', 'farm',
                                   'frozen_food', 'greengrocer', 'health_food',
                                   'ice_cream', 'organic', 'pasta', 'pastry',
                                   'seafood', 'spices', 'tea', 'water',
                                   'department_store', 'general', 'kiosk', 'mall',
                                   'supermarket', 'wholesale'])
                   }],
    'education_point': ['planet_osm_point',
                        {'amenity': db_set(['college', 'kindergarten', 'library',
                                            'school', 'university', 'research_institute',
                                            'music_school', 'language_school'])
                         }],
    'education_poly': ['planet_osm_polygon',
                       {'amenity': db_set(['college', 'kindergarten', 'library',
                                           'school', 'university', 'research_institute',
                                           'music_school', 'language_school']),
                        'building': db_set(['kindergarten', 'school', 'university'])
                        }],
    'school_point': ['planet_osm_point',
                     {'amenity': db_set(['college', 'kindergarten', 'library',
                                         'school', 'music_school', 'language_school'])
                      }],
    'school_poly': ['planet_osm_polygon',
                    {'amenity': db_set(['college', 'kindergarten', 'library',
                                        'school', 'music_school', 'language_school']),
                     'building': db_set(['kindergarten', 'school'])
                     }],
    'university_point': ['planet_osm_point',
                         {'amenity': db_set(['university', 'research_institute'])
                          }],
    'university_poly': ['planet_osm_polygon',
                        {'amenity': db_set(['university', 'research_institute']),
                         'building': db_set(['university'])
                         }],
    'transport_point': ['planet_osm_point',
                        {'amenity': db_set(['bus_station', 'car_rental', 'ferry_terminal']),
                         'public_transport': db_set(['station', 'platform'])
                         }],
    'transport_poly': ['planet_osm_polygon',
                       {'amenity': db_set(['bus_station', 'car_rental', 'ferry_terminal']),
                        'building': db_set(['train_station', 'transportation', 'parking']),
                        'public_transport': db_set(['station', 'platform'])
                        }],
    'health_point': ['planet_osm_point',
                     {'amenity': db_set(['clinic', 'dentist', 'doctors', 'hospital',
                                         'pharmacy', 'social_facility', 'veterinary'])
                      }],
    'health_poly': ['planet_osm_polygon',
                    {'amenity': db_set(['clinic', 'dentist', 'doctors', 'hospital',
                                        'pharmacy', 'social_facility', 'veterinary']),
                     'building': db_set(['hospital'])
                     }],
    'enterteinment_point': ['planet_osm_point',
                            {'amenity': db_set(['arts_centre', 'casino', 'cinema',
                                                'community_centre', 'nightclub', 'planetarium',
                                                'social_centre', 'theatre']),
                             'leisure': db_set(['stadium'])
                             }],
    'enterteinment_poly': ['planet_osm_polygon',
                           {'amenity': db_set(['arts_centre', 'casino', 'cinema',
                                               'community_centre', 'nightclub', 'planetarium',
                                               'social_centre', 'theatre']),
                            'leisure': db_set(['stadium']),
                            'building': db_set(['stadium'])
                            }],
    #     'big_entertainment_poly' : ['planet_osm_polygon',
    #                         {'leisure' : ['stadium']}],
    'retail_point': ['planet_osm_point',
                     {'shop': db_set(['alcohol', 'bakery', 'beverages',
                                      'brewing_supplies', 'butcher', 'cheese',
                                      'chocolate', 'coffee', 'confectionery',
                                      'convenience', 'deli', 'dairy', 'farm',
                                      'frozen_food', 'greengrocer', 'health_food',
                                      'ice_cream', 'organic', 'pasta', 'pastry',
                                      'seafood', 'spices', 'tea', 'water',
                                      'department_store', 'general', 'kiosk', 'mall',
                                      'supermarket', 'wholesale'], query_str='NOT IN'),
                      'amenity': db_set(['marketplace', 'post_office']),
                      'highway': db_set(['rest_area'])
                      }],
    'retail_poly': ['planet_osm_polygon',
                    {'shop': db_set(['alcohol', 'bakery', 'beverages',
                                     'brewing_supplies', 'butcher', 'cheese',
                                     'chocolate', 'coffee', 'confectionery',
                                     'convenience', 'deli', 'dairy', 'farm',
                                     'frozen_food', 'greengrocer', 'health_food',
                                     'ice_cream', 'organic', 'pasta', 'pastry',
                                     'seafood', 'spices', 'tea', 'water',
                                     'department_store', 'general', 'kiosk', 'mall',
                                     'supermarket', 'wholesale'], query_str='NOT IN'),
                     'amenity': db_set(['marketplace', 'post_office']),
                     'highway': db_set(['rest_area'])
                     }]
}


def query_feature(feature2query, within_poly, connection, feature2process_funcs=None, feature2db_table=None):
    """
    within_poly = "ST_GeographyFromText('%s')"%polytxt

    """
    featurename2cols2values = {}

    for name, (table, col2query_txt) in feature2query.items():

        featurename2cols2values[name] = {}

        for col, query_txt in col2query_txt.items():

            featurename2cols2values[name][col] = []
            col2vals = featurename2cols2values[name][col]

            if feature2db_table is not None:
                try:
                    table = feature2db_table[name]
                except KeyError:
                    pass

            #             if '_polygon' in table or '_line' in table:
            intersection = 'ST_Intersection(way, %s)' % within_poly
            query = """SELECT * """ \
                    """, {} AS intersection """ \
                    """FROM {} WHERE """ \
                    """ST_Intersects(way, {}) """ \
                    """AND {} {};""".format(intersection, table, within_poly, col, query_txt)

            #             elif '_point' in table:

            gdf = gpd.GeoDataFrame.from_postgis(query, connection, \
                                                geom_col='intersection', index_col='osm_id')
            gdf.drop('way', 1, inplace=True)
            gdf.rename(columns={'intersection': 'way'}, inplace=True)

            # process feature
            def put_zeroes(col2vals, name):
                if name[-5:] == '_poly':
                    col2vals += [0.0, 0]
                else:
                    col2vals += [0.0]
                return col2vals

            if len(gdf) == 0:
                col2vals = put_zeroes(col2vals, name)
            elif feature2process_funcs is not None:
                try:
                    for func in feature2process_funcs[name]:
                        col2vals += [func(gdf)]  # [gdf.apply(func)]
                except (KeyError, AttributeError):
                    #col2vals += [gdf]
                    col2vals = put_zeroes(col2vals, name)
            else:
                #col2vals += [gdf]
                col2vals = put_zeroes(col2vals, name)

    return featurename2cols2values


def aggregate_results(res):
    return list(np.sum(np.array(list(res.values())), axis=0))


def query_all_objects_in_bbox(xmin, ymin, xmax, ymax, connection, engine2):
    # polygons
    query = """SELECT * FROM planet_osm_polygon WHERE """ \
            """ST_Intersects(way, ST_MakeEnvelope({},{},{},{}, 4326)) ;""" \
        .format(xmin, ymin, xmax, ymax)
    gdf2 = gpd.GeoDataFrame.from_postgis(query, connection, geom_col='way', index_col='osm_id')

    gdf2['way'] = gdf2['way'].apply(lambda x: db.WKTElement(x.wkt, srid=4326))

    gdf2.to_sql('query_poly', engine2, index=True, if_exists='replace', \
                dtype={'way': db.Geometry('GEOMETRY', srid=4326), \
                       'tags': sqlalchemy.types.JSON})

    # lines
    query = """SELECT * FROM planet_osm_line WHERE """ \
            """ST_Intersects(way, ST_MakeEnvelope({},{},{},{}, 4326)) ;""" \
        .format(xmin, ymin, xmax, ymax)
    gdf2 = gpd.GeoDataFrame.from_postgis(query, connection, geom_col='way', index_col='osm_id')

    gdf2['way'] = gdf2['way'].apply(lambda x: db.WKTElement(x.wkt, srid=4326))

    gdf2.to_sql('query_line', engine2, index=True, if_exists='replace', \
                dtype={'way': db.Geometry('GEOMETRY', srid=4326), \
                       'tags': sqlalchemy.types.JSON})

    # points
    query = """SELECT * FROM planet_osm_point WHERE """ \
            """ST_Intersects(way, ST_MakeEnvelope({},{},{},{}, 4326)) ;""" \
        .format(xmin, ymin, xmax, ymax)
    gdf2 = gpd.GeoDataFrame.from_postgis(query, connection, geom_col='way', index_col='osm_id')

    gdf2['way'] = gdf2['way'].apply(lambda x: db.WKTElement(x.wkt, srid=4326))

    gdf2.to_sql('query_point', engine2, index=True, if_exists='replace', \
                dtype={'way': db.Geometry('POINT', srid=4326), \
                       'tags': sqlalchemy.types.JSON})


def isinside(lng_lat, bbox):
    try:
        lng, lat = lng_lat.xy
        lng, lat = lng[0], lat[0]
    except AttributeError:
        lng, lat = lng_lat
    xmin, ymin, xmax, ymax = bbox
    if (xmin < lng <= xmax) and (ymin < lat <= ymax):
        return True
    else:
        return False


def get_geom_centroid(geom, return_lat_lng=False):
    lng, lat = map(lambda x: x.pop(), geom.centroid.xy)
    if return_lat_lng:
        return [lat, lng]
    else:
        return [lng, lat]


def read_zipped_shp_to_gdf(zip_file, shp_file):
    """
    regione = 19
    shp_file = 'R%s_11_WGS84/R%s_11_WGS84.shp' % (regione, regione)
    zip_file = sez_dir + 'R%s_11_WGS84.zip' % regione

    """
    #with fiona.open(shp_file, vfs='zip://%s' % zip_file) as c:
    #    gdf = gpd.GeoDataFrame.from_features([feat for feat in c], crs=c.crs).to_crs(epsg=4326)
    #return gdf
    return gpd.read_file('zip://%s!%s'%(zip_file, shp_file)).to_crs("EPSG:4326")


def read_shp_ita_region(regione):
    shp_file = 'R%s_11_WGS84/R%s_11_WGS84.shp' % (regione, regione)
    zip_file = sez_dir + 'shp_sezioni/R%s_11_WGS84.zip' % regione
    gdf = read_zipped_shp_to_gdf(zip_file, shp_file)
    return gdf.loc[:, ['SEZ2011', 'geometry']]


england = {
    # 'read_shp': lambda: gpd.read_file(uk_dir + 'boundary_data/infuse_oa_lyr_2011_shp/'),
    'read_shp': (lambda: gpd.read_file(uk_dir + 'boundary_data/infuse_oa_lyr_2011_shp/') for _ in range(0)),
    'loc_ID_shp': 'geo_code',
    'db_name': 'osm',
    'tess_bbox': '../data/tess_bbox_all.csv',
    'output_dir': osm_dir + 'oa2handmade_features/'
}

ita = {
    'read_shp': (lambda: read_shp_ita_region(r) for r in [str(i).zfill(2) for i in range(1, 21)]),
    'loc_ID_shp': 'SEZ2011',
    'db_name': 'osm_ita',
    'tess_bbox': '../data/geo_data/tess_bbox_ita.csv',
    'output_dir': './'
}


def get_features(info_dict):

    # load OAs
    print("loading stuff")

    for read_shp_file in info_dict['read_shp']:
        oa_gdf = read_shp_file().astype({info_dict['loc_ID_shp']: 'str'})

        # Compute centroids, if not present
        if 'centroid' not in oa_gdf.columns:
            print("computing centroids")
            centr = oa_gdf.geometry.apply(get_geom_centroid, return_lat_lng=False)
            oa_gdf.loc[:, 'centroid'] = centr
        else:
            oa_gdf['centroid'] = oa_gdf['centroid'].apply(literal_eval)

        # connect to dbs
        db_name = info_dict['db_name']
        engine, connection, cursor = db.connect_to_db_sqlalchemy(database=db_name, user=USER)

        db.create_database('osm_queries', USER, spatial=True, private=False)
        engine2, connection2, cursor2 = db.connect_to_db_sqlalchemy('osm_queries', user=USER)

        # load tessellation
        tess_bbox = pd.read_csv(info_dict['tess_bbox'])

        # try to see if a copy of the Features dictionary is already present
        out_dir = info_dict['output_dir']
        try:
            with open(out_dir + 'bbox2oa2handmade_features.json', 'r') as f:
                    tileid2oa2features = json.load(f)
        except FileNotFoundError:
            print('%s not found.'%(out_dir + 'bbox2oa2handmade_features.json'))
            tileid2oa2features = {}

        len_bbox = len(tess_bbox)

        for ii, r in enumerate(tess_bbox.itertuples()) :
            xmin, ymin, xmax, ymax = r.xmin, r.ymin, r.xmax, r.ymax
            bbox = np.array([xmin, ymin, xmax, ymax])
            tile_id = str(r.tile_ID)

            # Comment the lines below when a tile may contain multiple shape files
            # # do not recompute features
            # if str(tile_id) in tileid2oa2features:
            #     continue

            sys.stdout.write(" bbox %s, tile id %s                                                                   \r" % (ii, tile_id))

            # Find all OAs within the bbox
            oas_within_bbox = oa_gdf[oa_gdf.centroid.apply(lambda x: isinside(x, bbox))]

            # do not process OAs already in tileid2oa2features
            if tile_id in tileid2oa2features:
                missing_oas = set(oas_within_bbox[info_dict['loc_ID_shp']].values) - \
                              set(list(tileid2oa2features[tile_id].keys()))
                oas_within_bbox = oas_within_bbox[oas_within_bbox[info_dict['loc_ID_shp']].isin(missing_oas)]

            tot = len(oas_within_bbox)
            if tot == 0:
                continue

            # select all OSM objects in the bbox and save to db
            sys.stdout.write(' querying all OSM objects within tile id %s (%s missing) ...                                           \r' % (tile_id, tot))
            query_all_objects_in_bbox(xmin, ymin, xmax, ymax, connection, engine2)

            oa2features = {}
            ddct = {c: feature2query[c] for c in feature2db_table.keys()}

            count = 1
            # for index, gdfoa in gdf_bristol.iterrows():
            for j, gdfoa in oas_within_bbox.iterrows():
                sys.stdout.write(' bbox %s of %s: %s of %s ...                                                         \r' % (ii + 1, len_bbox + 1, count, tot))

                oa = gdfoa[info_dict['loc_ID_shp']]

                # # do not process OAs already in tileid2oa2features
                # if str(oa) in tileid2oa2features[tile_id]:
                #     continue

                polytxt = gdfoa.geometry.wkt
                within_poly = """ST_GeographyFromText('%s')""" % polytxt

                # try:
                featurename2cols2values = query_feature(ddct, \
                                                        within_poly, connection2, \
                                                        feature2process_funcs=feature2process_funcs, \
                                                        feature2db_table=feature2db_table)
                # except KeyError:
                #     continue

                oa2features[oa] = {k: aggregate_results(v) for k, v in featurename2cols2values.items()}

                count += 1

            # tileid2oa2features[tile_id] = oa2features
            try:
                tileid2oa2features[tile_id] = {**tileid2oa2features[tile_id], **oa2features}
            except KeyError:
                tileid2oa2features[tile_id] = oa2features

            # save data processed every 100 bboxes
            if (ii + 1) % 100 == 0:
                with open(out_dir + 'bbox2oa2handmade_features.json', 'w') as f:
                    json.dump(tileid2oa2features, f, cls=NumpyEncoder)


        with open(out_dir + 'bbox2oa2handmade_features.json', 'w') as f:
            json.dump(tileid2oa2features, f, cls=NumpyEncoder)

        db.disconnect_db(cursor2, connection2, engine2)
        db.drop_database('osm_queries', cursor)
        print('\n')


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore')
    #warnings.resetwarnings()

    get_features(ita)

