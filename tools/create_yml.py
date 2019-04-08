#!env python

import os
import subprocess
import shutil
import hashlib
from ruamel.yaml import YAML

term_dict = {'ssd': 'SSD', 'rcnn': 'RCNN', 'resnet': 'ResNet', 'resnet50': 'ResNet50', 'resnet101': 'ResNet101',
             'nas': 'NAS', 'coco': 'COCO', 'coco14': 'COCO14', 'rfcn': 'RFCN', 'mobilenet': 'MobileNet',
             'ssdlite': 'SSDLite', 'fpn': 'FPN', 'ppn': 'PPN'}

# use md5sum for graph file checksum
hash_md5 = hashlib.md5()
yaml = YAML()
yaml.default_flow_style = False

# sample yml as the base and load the structures inside the sample_yml
yml_dir = os.path.join(os.getcwd(), "../builtin_models")
sample_complete_name = 'ssd_mobilenet_v2_coco_2018_03_29'
sample_pretty_name = 'SSD_MobileNet_v2_COCO'
sample_yml = os.path.join(yml_dir, sample_pretty_name + ".yml")
with open(sample_yml, 'r') as stream:
    yml_data = yaml.load(stream)

# Get the clean name of each model
model_paths = subprocess.check_output(
    "ls -d detectionModelZoo/*/", shell=True).decode("utf-8")
model_paths = model_paths.split('\n')[:-1]

model_names = []
pretty_names = []
for i, model_path in enumerate(model_paths):
    model_name = model_path.split("/")[1]
    model_names.append(model_name)
    terms = model_name.split("_")[:-3]
    for i in range(len(terms)):
        if terms[i] in term_dict.keys():
            terms[i] = term_dict[terms[i]]
        elif terms[i][0] == 'v':
            continue
        else:
            terms[i] = terms[i].capitalize()
    pretty_names.append('_'.join(terms))

# for i in range(len(model_paths)):
#     print(model_paths[i])
#     print(pretty_names[i])

last_model_name = sample_complete_name

for model_path, complete_name, pretty_name in zip(model_paths, model_names, pretty_names):
    # if the yml file already existed, continue to the next one
    new_yml = os.path.join(yml_dir, pretty_name + ".yml")
    if os.path.isfile(new_yml):
        print(pretty_name, "has already existed")
        continue
    else:
        print('Creating', pretty_name)

    # generate checksum with the model_path
    graph = os.path.join(model_path, 'frozen_inference_graph.pb')
    with open(graph, 'rb') as g:
        graph_bytes = g.read()
    checksum = hashlib.md5(graph_bytes).hexdigest()

    print(checksum)
    print(pretty_name)
    print(complete_name)

    # fill out the new yml file with model_name, pretty_name and checksum
    yml_data['name'] = pretty_name
    yml_data['description'] = yml_data['description'].replace(
        last_model_name, complete_name)
    yml_data['model']['graph_path'] = yml_data['model']['graph_path'].replace(
        last_model_name, complete_name)
    yml_data['model']['graph_checksum'] = checksum
    yml_data['attributes']['manifest_author'] = 'Jingning Tang'
    last_model_name = complete_name

    # Uncomment for sanity check
    # for key, item in yml_data.items():
    #     print(key, ":", item)

    with open(new_yml, 'w') as f:
        yaml.dump(yml_data, f)
