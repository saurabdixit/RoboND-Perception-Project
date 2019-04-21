#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import rospkg
import rosparam
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {'object_list' : dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    ##########################################################################
    # Exercise-2 TODOs:
    ##########################################################################

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.003
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()

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


    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(ros_cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(25)
    ec.set_MaxClusterSize(13000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

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

    # TODO: Convert PCL data to ROS messages
    # TODO: Publish ROS messages
    pcl_table_pub.publish(pcl_to_ros(ros_cloud_table))
    pcl_objects_pub.publish(pcl_to_ros(ros_cloud_objects))
    pcl_clustercloud_pub.publish(pcl_to_ros(cluster_cloud))

    ##########################################################################
    # Exercise-3 TODOs: 
    ##########################################################################
    
    
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

                    #print(type(test_scene_num.data))
                    #print(type(arm_name.data))
                    #print(type(obj_name.data))
                    #print(type(pick_pose.position.x), type(pick_pose.position.y), type(pick_pose.position.z))
                    #print(type(place_pose.position.x), type(place_pose.position.y), type(place_pose.position.z))
                    yaml_dict = make_yaml_dict(test_scene_num, arm_name, obj_name, pick_pose, place_pose)
                    dict_list.append(yaml_dict)
        
        print(dict_list)
        send_to_yaml(cur_pkg_path+'/config/output_data_world_'+str(test_scene_num.data)+'.yaml', dict_list)
        if len(dict_list) == len(object_list): 
            print("-------------------------------------")
            print("Writing output file using above data")
            print("-------------------------------------")
            is_yaml_created = True



    


        

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    #try:
    #    pr2_mover(detected_objects_list)
    #except rospy.ROSInterruptException:
    #    pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file

#Helper function to get test_scene_num
def GetTestSceneNum(LaunchFilePath):
    world_name = ''
    with open(launch_file_path,'rb') as launch_file:
        for line in launch_file:
            if 'world_name' in line:
                world_name = line

    test_scene_num = Int32()

    if 'test1' in world_name:
        test_scene_num.data = 1
    elif 'test2' in world_name:
        test_scene_num.data = 2
    elif 'test3' in world_name:
        test_scene_num.data = 3

    return test_scene_num


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('pr2_perception',anonymous = True)

    # IS YAML created?
    is_yaml_created = False

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers

    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_clustercloud_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_labels", Marker, queue_size = 10)
    detected_object_pub = rospy.Publisher("/detected_object_array", DetectedObjectsArray, queue_size = 10)

    # Load model from the disk
    rospack = rospkg.RosPack()
    cur_pkg_path = rospack.get_path('pr2_robot')

    #Get test_scene_num from the launch file
    launch_file_path = cur_pkg_path+'/launch/pick_place_project.launch'
    test_scene_num = GetTestSceneNum(launch_file_path)

    #Load correct pick list using test_scene_num
    pick_list_file = cur_pkg_path+'/config/pick_list_'+str(test_scene_num.data)+'.yaml' 
    paramlist = rosparam.load_file(pick_list_file)
    for params, ns in paramlist:
        rosparam.upload_params(ns, params)

    #Load drop poses
    drop_list = rosparam.get_param('/dropbox')
    
    #Get Saved Model
    svm_model_path = rospkg.RosPack()
    model = pickle.load(open(cur_pkg_path+'/svm/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']


    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
