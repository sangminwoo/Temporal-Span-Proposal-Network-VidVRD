# Video Visual Relationship Detection & Video Object Relation

## VidVRD
[vidvrd](https://xdshang.github.io/docs/imagenet-vidvrd.html)

### Preprocessing
use `vidvrd_to_image.sh`

### Annotations structure
The json file contains a dictionary sturctured like:
```
{
    "video_id": "ILSVRC2015_train_00010001",        # Video ID from the original ImageNet ILSVRC2016 video dataset
    "frame_count": 219,
    "fps": 30, 
    "width": 1920, 
    "height": 1080, 
    "subject/objects": [                            # List of subject/objects
        {
            "tid": 0,                               # Trajectory ID of a subject/object
            "category": "bicycle"
        }, 
        ...
    ], 
     "trajectories": [                              # List of frames
        [                                           # List of bounding boxes in each frame
            {
                "tid": 0,                       
                "bbox": {
                    "xmin": 672,                    # left
                    "ymin": 560,                    # top
                    "xmax": 781,                    # right
                    "ymax": 693                     # bottom
                }, 
                "generated": 0,                     # 0 - the bounding box is manually labeled
                                                    # 1 - the bounding box is automatically generated
            }, 
            ...
        ],
        ...
    ]
    "relation_instances": [                         # List of annotated visual relation instances
        {
            "subject_tid": 0,                       # Corresponding trajectory ID of the subject
            "object_tid": 1,                        # Corresponding trajectory ID of the object
            "predicate": "move_right", 
            "begin_fid": 0,                         # Frame index where this relation begins (inclusive)
            "end_fid": 210                          # Frame index where this relation ends (exclusive)
        }, 
        ...
    ]
}
```

## VidOR
[vidor](https://xdshang.github.io/docs/vidor.html)

### Preprocessing
use `vidor_to_image.sh`

### Annotations structure
The json file contains a dictionary sturctured like:
```
{
    "version": "VERSION 1.0",
    "video_id": "5159741010",                       # Video ID in YFCC100M collection
    "video_hash": "6c7a58bb458b271f2d9b45de63f3a2", # Video hash offically used for indexing in YFCC100M collection 
    "video_path": "1025/5159741010.mp4",            # Relative path name in this dataset
    "frame_count": 219,
    "fps": 29.97002997002997, 
    "width": 1920, 
    "height": 1080, 
    "subject/objects": [                            # List of subject/objects
        {
            "tid": 0,                               # Trajectory ID of a subject/object
            "category": "bicycle"
        }, 
        ...
    ], 
    "trajectories": [                               # List of frames
        [                                           # List of bounding boxes in each frame
            {                                       # The bounding box at the 1st frame
                "tid": 0,                           # The trajectory ID to which the bounding box belongs
                "bbox": {
                    "xmin": 672,                    # Left
                    "ymin": 560,                    # Top
                    "xmax": 781,                    # Right
                    "ymax": 693                     # Bottom
                }, 
                "generated": 0,                     # 0 - the bounding box is manually labeled
                                                    # 1 - the bounding box is automatically generated by a tracker
                "tracker": "none"                   # If generated=1, it is one of "linear", "kcf" and "mosse"
            }, 
            ...
        ],
        ...
    ],
    "relation_instances": [                         # List of annotated visual relation instances
        {
            "subject_tid": 0,                       # Corresponding trajectory ID of the subject
            "object_tid": 1,                        # Corresponding trajectory ID of the object
            "predicate": "in_front_of", 
            "begin_fid": 0,                         # Frame index where this relation begins (inclusive)
            "end_fid": 210                          # Frame index where this relation ends (exclusive)
        }, 
        ...
    ]
}
```