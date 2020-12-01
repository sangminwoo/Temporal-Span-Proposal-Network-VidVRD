#!/usr/bin/env bash

read_dir=~/data/vidor/video
save_dir=~/data/vidor/image

if [[ ! -f ${save_dir} ]]; then
	mkdir ${save_dir}
fi

cd ${read_dir}
for dir in *; do
	cd ${read_dir}/${dir}
	for sub_dir in *; do
		if [[ ! -f ${save_dir}/${sub_dir:0:-4} ]]; then
			mkdir ${save_dir}/${sub_dir:0:-4}
		fi
		ffmpeg -i ${read_dir}/${dir}/${sub_dir} ${save_dir}/${sub_dir:0:-4}/%05d.jpg
	done
done
