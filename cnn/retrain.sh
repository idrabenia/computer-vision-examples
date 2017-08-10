#!/usr/bin/sh

python retrain.py --model_dir=./model --output_graph=./model/output_graph.pb --image_dir=./Data --output_labels=./model/output_labels.txt
