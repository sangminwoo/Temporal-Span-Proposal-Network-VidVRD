#!/usr/bin/env bash

read_dir1=~/data/vidvrd/vidvrd-videos-part1
read_dir2=~/data/vidvrd/vidvrd-videos-part2
save_dir=~/data/vidvrd/image

if [[ ! -f ${save_dir} ]]; then
	mkdir ${save_dir}
fi

cd ${read_dir1}
for dir in *; do
	if [[ ! -f ${save_dir}/${dir:0:-4} ]]; then
		mkdir ${save_dir}/${dir:0:-4}
	fi
	ffmpeg -i ${read_dir1}/${dir} ${save_dir}/${dir:0:-4}/%05d.jpg
done

cd ${read_dir2}
for dir in *; do
	if [[ ! -f ${save_dir}/${dir:0:-4} ]]; then
		mkdir ${save_dir}/${dir:0:-4}
	fi
	ffmpeg -i ${read_dir2}/${dir} ${save_dir}/${dir:0:-4}/%05d.jpg
done
