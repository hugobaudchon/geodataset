import pandas as pd
import geopandas as gpd
import pygbif
import json


def get_gbif_id(taxon_name: str, rank: str, genus: str = None, family: str = None):
    if rank == 'species' and (genus is None or family is None):
        raise ValueError("genus and family must be specified when rank is 'species'.")
    elif rank == 'genus' and family is None:
        raise ValueError("family must be specified when rank is 'genus'.")

    search_results = pygbif.species.name_lookup(q=taxon_name, rank=[rank], limit=10)

    if taxon_name == 'Populus':
        print(search_results)

    if search_results['results']:
        key = None
        if rank == 'species':
            for result in search_results['results']:
                if 'genus' not in result or 'family' not in result:
                    if result == search_results['results'][-1]:
                        key = result['nubKey']
                        break
                    else:
                        continue

                if result['genus'] == genus and result['family'] == family:
                    key = result['nubKey']
                    break
        elif rank == 'genus':
            for result in search_results['results']:
                if 'family' not in result:
                    if result == search_results['results'][-1]:
                        key = result['nubKey']
                        break
                    else:
                        continue
                if result['family'] == family:
                    key = result['nubKey']
                    break
        elif rank == 'family':
            key = search_results['results'][0]['nubKey']

        return key
    else:
        return None


def gdf_to_coco_categories(gdf: gpd.GeoDataFrame):
    assert all(level in gdf.columns for level in ['canonicalName', 'species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom'])

    coco_categories_family = []
    category_id = 1

    print("Getting families...")
    families = gdf['family'].unique()
    for family in families:
        if not family:
            continue

        coco_categories_family.append({
          "id": category_id,
          "name": family,
          "global_id": int(get_gbif_id(family, rank='family')),
          "rank": "FAMILY",
          "other_names": [],
          "supercategory": None
        })

        print(f"Found family {family}")

        category_id += 1

    coco_categories_genus = []
    print("Getting genuses...")
    for coco_family in coco_categories_family:
        family = coco_family['name']
        family_df = gdf[gdf['family'] == family]

        for genus in family_df['genus'].unique():
            if not genus:
                continue

            coco_categories_genus.append({
                "id": category_id,
                "name": genus,
                "global_id": int(get_gbif_id(genus, rank='genus', family=family)),
                "rank": "GENUS",
                "other_names": [],
                "supercategory": coco_family['id']
            })

            print(f"Found genus {genus} in family {family}")

            category_id += 1

    coco_categories_species = []
    print("Getting species...")
    for coco_genus in coco_categories_genus:
        genus = coco_genus['name']
        genus_df = gdf[gdf['genus'] == genus]

        family = gdf[gdf['genus'] == genus]['family'].values[0]

        for species in genus_df['species'].unique():
            if not species:
                continue

            coco_categories_species.append({
                "id": category_id,
                "name": species,
                "global_id": int(get_gbif_id(species, rank='species', genus=genus, family=family)),
                "rank": "SPECIES",
                "other_names": [],
                "supercategory": coco_genus['id']
            })

            print(f"Found species {species} in genus {genus}")

            category_id += 1

    coco_categories = coco_categories_family + coco_categories_genus + coco_categories_species

    return coco_categories


if __name__ == '__main__':
    gdfs_paths = [
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20170810_transectotoni_mavicpro_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20230525_tbslake_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_inundated_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_pantano_m3e_labels_points.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/labels_points_species/20231018_terrafirme_m3e_labels_points.gpkg',
    ]
    #
    output_file = 'equator_tiputini_trees/equator_tiputini_trees_categories.json'
    #
    gdfs = [gpd.read_file(gdf_path) for gdf_path in gdfs_paths]

    # to same crs
    gdfs = [gdf.to_crs(gdfs[0].crs) for gdf in gdfs]

    gdf = gpd.GeoDataFrame(pd.concat(gdfs))

    # gdf = gpd.GeoDataFrame(pd.read_csv('/home/hugobaudchon/Downloads/sbl_cloutier_taxa_gbif.csv'))
    # gdf['species'] = gdf['species'].apply(lambda x: x if pd.notna(x) else None)
    # gdf['genus'] = gdf['genus'].apply(lambda x: x if pd.notna(x) else None)
    # gdf['family'] = gdf['family'].apply(lambda x: x if pd.notna(x) else None)

    coco_categories = gdf_to_coco_categories(gdf)

    with open(output_file, 'w') as f:
        json.dump({"categories": coco_categories}, f, indent=4)
