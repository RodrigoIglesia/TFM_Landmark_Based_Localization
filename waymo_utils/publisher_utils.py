import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import struct

def publish_incremental_pose_to_topic(topic, pose, text, header):
    """
    Publish an incremental odometry message and a text marker in RVIZ.

    Parameters:
        topic (str): The ROS topic to publish the messages to.
        pose (list or tuple): Accumulated pose as [x, y, z, roll, pitch, yaw].
        text (str): Text to display in the marker.
        header (str): Frame ID for the pose and marker messages.
    Returns:
    - None
    """

    # Set up publishers
    vehicle_pose_pub = rospy.Publisher(topic + "/odometry", Odometry, queue_size=10)
    pose_text_pub = rospy.Publisher(topic + "/text_marker", Marker, queue_size=10)

    # Odometry message for incremental pose
    incremental_pose_msg = Odometry()
    incremental_pose_msg.header = header
    incremental_pose_msg.pose.pose.position = Point(*pose[:3])

    # Convert Euler angles to quaternion for ROS message
    rotation = R.from_euler('xyz', pose[3:])
    quaternion = rotation.as_quat()
    incremental_pose_msg.pose.pose.orientation = Quaternion(*quaternion)
    vehicle_pose_pub.publish(incremental_pose_msg)

    # Text marker for pose information
    text_marker = Marker()
    text_marker.header = header
    text_marker.ns = "pose_text"
    text_marker.id = 0
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position = Point(pose[0], pose[1], pose[2] + 1.0)  # Slightly above the vehicle
    text_marker.pose.orientation = Quaternion(*quaternion)
    text_marker.scale.z = 0.5  # Height of the text
    text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # White color
    text_marker.text = text

    # Publish the marker
    pose_text_pub.publish(text_marker)
    rospy.loginfo("New incremental pose published")

def publish_multiple_poses_to_topic(topic, labeled_poses, header):
    """
    Publish multiple poses with labels to a ROS topic, where each label has an associated color and pose.

    Parameters:
        topic (str): The ROS topic to publish the messages to.
        labeled_poses (dict): Dictionary with labels as RGB lists (in float format) as keys and poses as values.
                              Each pose should be in the format [x, y, z, roll, pitch, yaw].
        header (Header): Header object with frame ID and timestamp.
    Returns:
    - None
    """

    # Set up publishers for individual markers
    vehicle_pose_pub = rospy.Publisher(topic + "/odometry", Odometry, queue_size=10)
    pose_text_pub = rospy.Publisher(topic + "/text_marker", Marker, queue_size=10)  # Single marker publisher

    # Loop through each label and pose in the dictionary
    for idx, (rgb_color, pose) in enumerate(labeled_poses.items()):
        # Odometry message for each pose
        pose_msg = Odometry()
        pose_msg.header = header  # Use full Header object directly
        pose_msg.pose.pose.position = Point(*pose[:3])

        # Convert Euler angles to quaternion for ROS message
        rotation = R.from_euler('xyz', pose[3:])
        quaternion = rotation.as_quat()
        pose_msg.pose.pose.orientation = Quaternion(*quaternion)

        # Publish the odometry message
        vehicle_pose_pub.publish(pose_msg)

        # Text marker for pose information
        text_marker = Marker()
        text_marker.header = header  # Use full Header object directly
        text_marker.ns = "pose_text"
        text_marker.id = idx  # Unique ID for each marker
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = Point(pose[0], pose[1], pose[2] + 1.0)  # Slightly above the position
        text_marker.pose.orientation = Quaternion(*quaternion)
        text_marker.scale.z = 0.5  # Height of the text

        # Convert RGB color from float list format (e.g., [0.5, 0.2, 0.7]) to ColorRGBA
        r, g, b = rgb_color
        text_marker.color = ColorRGBA(r, g, b, 1.0)  # Full opacity

        # Set the label text
        text_marker.text = f"Pose {idx}"

        # Publish the marker individually
        pose_text_pub.publish(text_marker)

    rospy.loginfo("Published multiple labeled poses to topic")


def publish_image_to_topic(topic, image, header):
    """
    Publishes an image to the specified ROS topic.

    param topic: The ROS topic to publish the image to (string).
    param image: The image to be published (OpenCV format).
    param header: The header to maintain the metadata (ROS header).

    Returns:
    - None
    """
    # Initialize the publisher for the specified topic
    image_publisher = rospy.Publisher(topic, Image, queue_size=10)

    # Convert OpenCV image to ROS Image message
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")

    # Maintain the header information from the original image
    image_msg.header = header

    # Publish the image message
    image_publisher.publish(image_msg)

def publish_labeled_pointcloud_to_topic(topic, clustered_pointcloud, header):
    """
    Publishes labeled point clouds from clustered point cloud data.

    Parameters:
    - topic: The ros topic where the pointcloud is published
    - clustered_pointcloud: A dictionary with labels as keys and points as values.
    - header: The header information for the PointCloud2 message.

    Returns:
    - None
    """
    # Initialize the publisher
    pointlcoud_publisher = rospy.Publisher(topic, PointCloud2, queue_size=10)
    # Initialize the filtered point cloud message
    final_pointcloud = []

    # Gather points from the clustered point clouds
    for rgb_color, points in clustered_pointcloud.items():
        # Assume rgb_color is a tuple (R, G, B) which could be in float range [0, 1] or int range [0, 255]
        # Scale and convert to integer if necessary
        r, g, b = [int(c * 255) if isinstance(c, float) and 0 <= c <= 1 else int(c) for c in rgb_color]
        # Encode the RGB color as a single float
        rgb_float = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        for point in points:
            x, y, z = point[:3]
            final_pointcloud.append([x, y, z, rgb_float])

    # Define the point fields
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1) 
    ]

    # Create the PointCloud2 message
    filtered_pointcloud_msg = pc2.create_cloud(header, fields, final_pointcloud)

    # Publish the filtered point cloud message
    pointlcoud_publisher.publish(filtered_pointcloud_msg)