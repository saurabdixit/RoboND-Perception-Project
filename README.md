# Project: Perception Pick & Place
Welcome to the implementation of Perception Pick and Place project!

[//]: # (Image References)
[world1]: ./misc/images/testworld1.png
[world2]: ./misc/images/testWorld2.png
[world3]: ./misc/images/testWorld3.png
[confusionmatrix]: ./misc/images/confusionMatrix.png
[ClusterDemo]: ./misc/images/Clusterdemo.png

### Test world 1
![][world1]
---

### Test world 2
![][world2]
---

### Test world 3
![][world3]
---


## Steps to run the project:
1. Clone the repository to the source folder of your catkin workspace. Note that this package is dependent on the sensor_stick package. Hence, make sure that you have the sensor_stick package built.
```bash
cd ~/catkin_ws/src/
git clone https://github.com/saurabdixit/RoboND-Perception-Project.git
```
2. If you have RoboND-Kinematics-Project in your catkin_ws/src, use following command to ignore build on that package as we have some repeated package name in RoboND-Perception-Project.
```bash
cd ~/catkin_ws/src/RoboND-Kinematics-Project/
touch CATKIN_IGNORE
```
3. Run following commands on your catkin workspace.
```bash
cd ~/catkin_ws/
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
catkin_make
```
4. Open pick_place_project.launch file to edit.
```bash
rosed pr2_robot pick_place_project.launch
```
5. Change following line to open the test\#.world of your choice.
```xml
 <!--For world 1 use following line-->
 <arg name="world_name" value="$(find pr2_robot)/worlds/test1.world"/>
 <!--For world 2 use following line-->
 <arg name="world_name" value="$(find pr2_robot)/worlds/test2.world"/>
 <!--For world 3 use following line-->
 <arg name="world_name" value="$(find pr2_robot)/worlds/test3.world"/>
```
6. No need to select the picklist. I am parsing above launch file to select the picklist.
7. SVM model is saved in following location. Hence, you are ready to run the project.
```bash
roscd pr2_robot/svm
```
8. Run following command, enable point cloud, and markers in rviz window to see the object detection
```bash
roslaunch pr2_robot pick_place_project.launch
```
9. output_data_world\_\#.yaml will be saved to following location
```bash
roscd pr2_robot/config
```
10. Thanks for running the project


## Following section covers the details of how the object recognition is setup.
All the exercises and the procedure to output yaml file has been clearly marked in the code. For more details, please open following code:
```bash
rosed pr2_robot pr2_perception.py
```
### Exercise 1, 2, and 3:
#### Exercise 1: Filtering and RANSAC plane fitting
Let's go through the steps one by one
* Initially, we need to convert the format to PCL to use pcl library functions
```python
    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
```
* Downsample the point cloud to something that we could process. The following LEAF_SIZE might be small for real-time processing. 
```python
    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.003
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()
```
* Set PassThrough filter. I have removed the table sides by setting the z axis_min to 0.6. Also, I have added a y-axis filter to remove corners of the drop boxes from point cloud data. 
```python
    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    #Making anothe y axis filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
```
* Find the table top and remove it from the object's point cloud. 
```python
    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    inliers, coefficients = seg.segment()

    ros_cloud_table = cloud_filtered.extract(inliers, negative=False)
    ros_cloud_objects = cloud_filtered.extract(inliers, negative=True)
    outlier_filter = ros_cloud_objects.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(7)
    x = 0.001
    outlier_filter.set_std_dev_mul_thresh(x)
    ros_cloud_objects = outlier_filter.filter()
```

#### Exercise 2: Clustering for segmentation
* Identify Euclidean clusters
```python
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(ros_cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(25)
    ec.set_MaxClusterSize(13000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
```
* Mark different clusters with different colors for visualization
```python

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
```
* Publish point cloud in ROS format
```python
    # TODO: Convert PCL data to ROS messages
    # TODO: Publish ROS messages
    pcl_table_pub.publish(pcl_to_ros(ros_cloud_table))
    pcl_objects_pub.publish(pcl_to_ros(ros_cloud_objects))
    pcl_clustercloud_pub.publish(pcl_to_ros(cluster_cloud))

```

![][ClusterDemo]


#### Exercise 3: Features extraction, SVM training, and object recognition
* compute_color_histograms() implementation
```python
def compute_color_histograms(cloud, using_hsv=False):
    point_colors_list = []
    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    # TODO: Compute histograms
    c1_hist = np.histogram(channel_1_vals, bins=32, range=(0,256))
    c2_hist = np.histogram(channel_2_vals, bins=32, range=(0,256))
    c3_hist = np.histogram(channel_3_vals, bins=32, range=(0,256))


    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0])).astype(np.float64)
    #print(hist_features)
    normed_features = hist_features / np.sum(hist_features)
    #print(normed_features)
    return normed_features 
```
* compute_normal_histograms() implementation
```python
def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    # TODO: Compute histograms
    c1_hist = np.histogram(channel_1_vals, bins=32, range=(0,256))
    c2_hist = np.histogram(channel_2_vals, bins=32, range=(0,256))
    c3_hist = np.histogram(channel_3_vals, bins=32, range=(0,256))

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0])).astype(np.float64)
    #print(hist_features)
    normed_features = hist_features / np.sum(hist_features)
    #print(normed_features)
    return normed_features 
```

I got 98% accuracy when I ran the train_svm.py for 25 iteration using hsv instead of rgb

![][confusionmatrix]

---


* Object recognition implementation: I am running prediction using hsv. Here is the implementation of Exercise 3 in pr2_perception.py
```python
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, points in enumerate(cluster_indices):
        # Grab the points for the cluster
        cloud_object = ros_cloud_objects.extract(points)

        # Compute the associated feature vector
        chists = compute_color_histograms(pcl_to_ros(cloud_object), using_hsv = True)
        normals = get_normals(pcl_to_ros(cloud_object))
        nhists = compute_normal_histograms(normals)

        feature_vector = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature_vector.reshape([1,-1])))

        # Publish a label into RViz
        label = encoder.inverse_transform(prediction)[0]

        # Add the detected object to the list of detected objects.
        detected_objects_labels.append(label)

        # Publish the list of detected objects
        label_pos = list(white_cloud[points[0]])
        label_pos[2] += 0.4
        object_markers_pub.publish(make_label(label, label_pos, index))

        do = DetectedObject()
        do.label = label
        do.cloud = pcl_to_ros(cloud_object)
        detected_objects.append(do)

    detected_object_pub.publish(detected_objects)
```

### Pick and Place Setup:
Following code from pr2_perception.py writes the output_test_world\#.yaml file as soon as it detects all the objects. The files will be stored in pr2_robot\config folder.
```python
global is_yaml_created
if not is_yaml_created:
    dict_list = []
    object_list = rosparam.get_param('/object_list')

    for obj in detected_objects:
        points_arr = ros_to_pcl(obj.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)

        for i in range(0, len(object_list)):
            if obj.label == object_list[i]['name']:
                pick_pose = Pose()
                pick_pose.position.x = np.asscalar(centroid[0])
                pick_pose.position.y = np.asscalar(centroid[1])
                pick_pose.position.z = np.asscalar(centroid[2])


                # Object name
                obj_name = String()
                obj_name.data = object_list[i]['name']

                # Arm to use
                arm_name = String()

                place_pose = Pose()
                if object_list[i]['group'] == 'red':
                    place_pose.position.x   = np.asscalar(np.float64(drop_list[0]['position'][0]))
                    place_pose.position.y   = np.asscalar(np.float64(drop_list[0]['position'][1]))
                    place_pose.position.z   = np.asscalar(np.float64(drop_list[0]['position'][2]))
                    arm_name.data = drop_list[0]['name']
                else:
                    place_pose.position.x   = np.asscalar(np.float64(drop_list[1]['position'][0]))
                    place_pose.position.y   = np.asscalar(np.float64(drop_list[1]['position'][1]))
                    place_pose.position.z   = np.asscalar(np.float64(drop_list[1]['position'][2]))
                    arm_name.data = drop_list[1]['name']


                yaml_dict = make_yaml_dict(test_scene_num, arm_name, obj_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)
    
    print(dict_list)
    send_to_yaml(cur_pkg_path+'/config/output_data_world_'+str(test_scene_num.data)+'.yaml', dict_list)
    if len(dict_list) == len(object_list): 
        print("-------------------------------------")
        print("Writing output file using above data")
        print("-------------------------------------")
        is_yaml_created = True
```

### Improvements:
* Need some parameter tuning to better detect the book object
* To identify objects in real-time, we need to make sure not to process too dense point cloud.
* We could make a ros parameter to launch the code using following command line argument instead of opening the file for changing the world_name.
```bash
roslaunch pr2_robot pick_place_project.launch test1.world
```


