import json


def merge_categories(main_coco_categories, additional_coco_categories):
    main_categories_mapping = {category['name']: category for category in main_coco_categories}
    next_category_id = max([category['id'] for category in main_coco_categories]) + 1

    old_id_to_new_id = {}
    for category in additional_coco_categories:
        if category['rank'].lower() != 'family':
            continue
        if category['name'] not in main_categories_mapping:
            old_id_to_new_id[category['id']] = next_category_id
            category['id'] = next_category_id
            next_category_id += 1
            main_coco_categories.append(category)
            main_categories_mapping[category['name']] = category
        else:
            old_id_to_new_id[category['id']] = main_categories_mapping[category['name']]['id']

    for category in additional_coco_categories:
        if category['rank'].lower() != 'genus':
            continue
        if category['name'] not in main_categories_mapping:
            old_id_to_new_id[category['id']] = next_category_id
            category['id'] = next_category_id
            category['supercategory'] = old_id_to_new_id[category['supercategory']] if category['supercategory'] in old_id_to_new_id else category['supercategory']
            next_category_id += 1
            main_coco_categories.append(category)
            main_categories_mapping[category['name']] = category
        else:
            old_id_to_new_id[category['id']] = main_categories_mapping[category['name']]['id']

    print(old_id_to_new_id)

    for category in additional_coco_categories:
        if category['rank'].lower() != 'species':
            continue
        if category['name'] not in main_categories_mapping:
            old_id_to_new_id[category['id']] = next_category_id
            category['id'] = next_category_id
            category['supercategory'] = old_id_to_new_id[category['supercategory']] if category['supercategory'] in old_id_to_new_id else category['supercategory']
            next_category_id += 1
            main_coco_categories.append(category)
            main_categories_mapping[category['name']] = category

    return main_coco_categories


if __name__ == '__main__':
    main_coco_categories = json.load(open('/home/hugo/PycharmProjects/geodataset/geodataset/utils/categories/brazil_zf2_trees/brazil_zf2_trees_categories.json'))['categories']
    additional_coco_categories = json.load(open('/home/hugo/PycharmProjects/geodataset/geodataset/utils/categories/brazil_zf2_trees/brazil_zf2_trees_categories_fornewzf2dataonly.json'))['categories']

    merged_coco_categories = merge_categories(main_coco_categories, additional_coco_categories)

    merged_coco_categories_json = {
        'categories': merged_coco_categories
    }

    with open('brazil_zf2_trees/brazil_zf2_trees_categories.json', 'w') as f:
        json.dump(merged_coco_categories_json, f, indent=4)


