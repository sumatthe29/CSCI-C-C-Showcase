"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import time

# Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
FINGER_LEFT = 12
FINGER_RIGHT = 13
N_PARTS = 14
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
# LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# ADDED FOR ARM
###########################
ARM_1 = 3
ARM_2 = 4
ARM_3 = 5
ARM_4 = 6
ARM_5 = 7
ARM_6 = 8
ARM_7 = 9
###########################

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

# Older robot_parts call, no idea where it is from or what it was for but left in just in case
# robot_parts={}

# for i, part_name in enumerate(part_names):
#     robot_parts[part_name]=robot.getDevice(part_name)
#     robot_parts[part_name].setPosition(float(target_pos[i]))
#     robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

robot_parts = []

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable Keyboard
# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = -5
pose_y     = 0
pose_theta = 0.876

goal_pose_x = -3.6  
goal_pose_y = 2.84
goal_pose_theta = 0

#Goal position
gpa = [[1, 2.2, 0]]

propor_number = 0

x_map = 0
y_map = 0

vL = 0
vR = 0

p1 = 1
p2 = 1
p3 = 0

# global variables for align mode
lane = 1
turn_counter = 0
where_is_object = ""  #left or right

# color blob detection variables

color_ranges = []
count = 0
goal_objects = set()
goal_positions = []
goal_positions_world = []

# List to hold sensor readings
lidar_sensor_readings = [] 
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)

lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# Stop Range Array to tell robot reorientation was sucessfull
stop_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160]


# ------------------------------------------------------------------
# Helper Functions

def add_color_range_to_detect(lower_bound, upper_bound):
  '''
  @param lower_bound: Tuple of BGR values
  @param upper_bound: Tuple of BGR values
  '''
  global color_ranges
  color_ranges.append([lower_bound, upper_bound]) # Add color range to global list of color ranges to detect

def check_if_color_in_range(bgr_tuple):
  '''
  @param bgr_tuple: Tuple of BGR values
  @returns Boolean: True if bgr_tuple is in any of the color ranges specified in color_ranges
  '''
  global color_ranges
  for entry in color_ranges:
    lower, upper = entry[0], entry[1]
    in_range = True
    for i in range(3):
      if bgr_tuple[i] < lower[i] or bgr_tuple[i] > upper[i]:
        in_range = False
        break
    if in_range: return True
  return False

def do_color_filtering(img):
  img_height = img.shape[0]
  img_width = img.shape[1]
  # Create a matrix of dimensions [height, width] using numpy
  mask = np.zeros([img_height, img_width]) # Index mask as [height, width] (e.g.,: mask[y,x])
  for y in range(img_height):
      for x in range(img_width):
        if check_if_color_in_range(img[y][x]):
          mask[y][x] = 1
  return mask

def expand_nr(img_mask, cur_coord, coordinates_in_blob):
  coordinates_in_blob = []
  coordinate_list = [cur_coord] # List of all coordinates to try expanding to
  while len(coordinate_list) > 0:
    cur_coordinate = coordinate_list.pop() # Take the first coordinate in the list and perform 'expand' on it
    if cur_coordinate[0] < 0 or cur_coordinate[1] < 0 or cur_coordinate[0] >= img_mask.shape[0] or cur_coordinate[1] >= img_mask.shape[1]: 
      continue
    if img_mask[cur_coordinate[0], cur_coordinate[1]] == 0.0: 
      continue

    img_mask[cur_coordinate[0],cur_coordinate[1]] = 0
    coordinates_in_blob.append(cur_coordinate)

    # TODO: Add all neighboring coordinates (above, below, left, right) to coordinate_list to expand to them
    coordinate_list.append([cur_coordinate[0]+1, cur_coordinate[1]])
    coordinate_list.append([cur_coordinate[0]-1, cur_coordinate[1]])
    coordinate_list.append([cur_coordinate[0], cur_coordinate[1]+1]) 
    coordinate_list.append([cur_coordinate[0], cur_coordinate[1]-1])

  return coordinates_in_blob

def get_blobs(img_mask):
  img_mask_height = img_mask.shape[0]
  img_mask_width = img_mask.shape[1]
  blobs_list = [] # List of all blobs, each element being a list of coordinates belonging to each blob
  img_copy = img_mask.copy()
  for y in range(img_mask_height):
        for x in range(img_mask_width):
            if img_mask[y][x] == 1:
                blobs_coords = expand_nr(img_copy, (y,x), [])
                blobs_list.append(blobs_coords)
  return blobs_list

def get_blob_centroids(blobs_list):
  object_positions_list = []
  for my_list_var in blobs_list:
      if len(my_list_var) > 0:
        centroid = np.mean(my_list_var, axis=0)
        object_positions_list.append(centroid)
  return object_positions_list

def see_yellow(count):
    # ##########################################################
    # #                 Color Blob Detection                   #
    # ##########################################################
    
    # blob detection to detect yellow objectts
    if count % 200 == 0:
        img = camera.getImage()
        img = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        # cv2.imwrite('camera_image.jpg', img)
        add_color_range_to_detect([0,150,200], [50,255,255])
        img_mask = do_color_filtering(img)
        blobs = get_blobs(img_mask)
        object_positions_list = get_blob_centroids(blobs)

        # call this whenever object_positions_list is not empty
        # and get locations of yellow objects
        if len(object_positions_list) > 0:
            objects = camera.getRecognitionObjects()
            for obj in objects:
                color = obj.getColors()
                obj_id = obj.getId()
                if obj_id not in goal_objects and color == [1.0, 1.0, 0.0]:
                    goal_positions.append(obj.getPosition())
                    goal_objects.add(obj_id)


# State Definitions 

mode = 'auto_start'                         # Mode defines what current operation is occuring with the robot
sub_mode = 'start'                              # sub_mode defines operations for automated mapping/color matching mode
position_counter = 0
gripper_status="closed"                     # Gripper Statues gives the status of the gripper for the arm

# mode = 'turn left 2'                         # Mode defines what current operation is occuring with the robot
# sub_mode = 'turn_left_2'                              # sub_mode defines operations for automated mapping/color matching mode
# position_counter = 2
# gripper_status="closed"                     # Gripper Statues gives the status of the gripper for the arm

# 12.9 1.97
# #####################################################
# #                     Mapping                       #
# #####################################################

# Map Initialization
map = np.zeros((360,360))

def map_func():

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 30:
            wx = 29.999
        # elif wx <= -30:
        #     wx = -29.999
        if wy >= 30:
            wy = 29.999
        # elif wy <= -16:
        #     wy = -15.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            # mapx = int(x_map - (wx*12))-60
            # mapy = int(y_map + (wy*12))

            mapx = int(180 - (wy*12))
            mapy = int(192 - (wx*12))

            # fixes oob error since the simulation sometimes throws values > index that break manual mode

            if (mapx >= 360):
                mapx = 359
            if (mapy >= 360):
                mapy = 359

            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            # if map[mapx][mapy] != 1: #

            # increase probability of obstacle exists
            map[mapx][mapy] += 0.025 # or 0.01    

            # prob of obstacle
            g = map[mapx][mapy] 

            if (g > 1): g = 1
            
            # Convert color to grey scale
            color = (g*256**2+g*256+g)*255 

            display.setColor(int(color))
            display.drawPixel(mapx, mapy)

# Main Loop
while robot.step(timestep) != -1:

    # #####################################################
    # #          Robot Coordinates PoseX/Y (GPS)          #
    # #####################################################

    # NOTE: Could not use existing odometry from lab5 since it was found to have problems.
    # An implementation is included but may or may not factor into actual operation

    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    n = compass.getValues()
    # print(n)
    rad = -((math.atan2(n[0], n[0]))-(1.5708*1.5))
    pose_theta = rad

    x_map = int(180 - (pose_y*12))
    y_map = int(192 - (pose_x*12))

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))

    display.drawPixel(x_map,y_map)

    # #####################################################
    # #               Arm Mechanics (TBD)                 #
    # #####################################################

    ##########################################################################
    # Grabbing stuff off the top shelf 
    # Set mode to grab0
    # Works for: X_robot = X_item and Y_robot = Y_item - 1.16
    # IE this works when the robot is aligned with the item on the x axis and
    # is 1.16 webots coordinates away from it in the y direction
    ##########################################################################

    if(mode == "grab0"):
        grab_counter = grab_counter + 1 #Grab_counter just serves as a delay so that
        #the arm can finish certain movements before beginning the next movement.

        #This first delay is so that the robot can become fully erect before moving the arm 
        if grab_counter >= 200:
            mode = "grab1"
            grab_counter = 0

    if(mode == "grab1"): #grab1 manipulates a couple of joints, then runs a delay before moving the next joints. This continues throughout the state machine

        robot_parts[FINGER_LEFT].setPosition(0.045)
        robot_parts[FINGER_LEFT].setPosition(0.045)

        robot_parts[ARM_2].setPosition(0.1)
        robot_parts[ARM_1].setPosition(1.6)

        grab_counter = grab_counter + 1
        if grab_counter >= 100:
            mode = "grab2"
            grab_counter = 0

    if(mode == "grab2"):

        robot_parts[ARM_4].setPosition(-0.1)
        robot_parts[ARM_5].setPosition(0)
        grab_counter = grab_counter + 1
        if grab_counter >= 100:
            mode = "grab3"
            grab_counter = 0

    if(mode == "grab3"):

        robot_parts[MOTOR_LEFT].setPosition(1)
        robot_parts[MOTOR_RIGHT].setPosition(1)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "grab4"
            grab_counter = 0

    if(mode == "grab4"):

        robot_parts[FINGER_LEFT].setPosition(0)
        robot_parts[FINGER_LEFT].setPosition(0)
        grab_counter = grab_counter + 1
        if grab_counter >= 45:
            mode = "grab5"
            grab_counter = 0


    if(mode == "grab5"):

        robot_parts[ARM_2].setPosition(0.3)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "grab6"
            grab_counter = 0

    if(mode == "grab6"):

        robot_parts[ARM_3].setPosition(1.5)
        grab_counter = grab_counter + 1
        if grab_counter >= 100:
            mode = "grab7"
            grab_counter = 0

    if(mode == "grab7"):

        robot_parts[ARM_6].setPosition(-1.39)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "grab8"
            grab_counter = 0

    if(mode == "grab8"):

        robot_parts[ARM_4].setPosition(2)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "grab9"
            grab_counter = 0

    if(mode == "grab9"):

        robot_parts[ARM_5].setPosition(2)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "grab10"
            grab_counter = 0

    if(mode == "grab10"):

        robot_parts[FINGER_LEFT].setPosition(0.045)
        robot_parts[FINGER_LEFT].setPosition(0.045)
        grab_counter = grab_counter + 1
        if grab_counter >= 50:
            mode = "grab11"
            grab_counter = 0

    if(mode == "grab11"): #Return arm to starting position
        for i in range(N_PARTS):
            robot_parts[i].setPosition(float(target_pos[i]))
        grab_counter = grab_counter + 1
        if grab_counter >= 200:
            mode = "manual"
            grab_counter = 0

    ############################################################################
    # Grabbing stuff off the BOTTOM shelf 
    # Set mode to lowgrab0
    # Works for: X_robot = X_item and Y_robot = Y_item - 1.06
    # IE this works when the robot is aligned with the item on the x axis and
    # is 1.06 webots coordinates away from it in the y direction
    ############################################################################

    if(mode == "lowgrab0"):
        grab_counter = grab_counter + 1 #Grab counter just serves as a delay so that
        #the arm can finish certain movements before beginning the next movement.

        #This first delay is so that the robot can become fully erect before moving the arm 
        if grab_counter >= 200:
            mode = "lowgrab1"
            grab_counter = 0

    if(mode == "lowgrab1"):

        robot_parts[FINGER_LEFT].setPosition(0.045)#open fingers
        robot_parts[FINGER_LEFT].setPosition(0.045)

        robot_parts[ARM_2].setPosition(-1.5)
        robot_parts[ARM_1].setPosition(1.6)
        robot_parts[ARM_4].setPosition(2.29)
        
        grab_counter = grab_counter + 1
        if grab_counter >= 100:
            mode = "lowgrab2"
            grab_counter = 0    

    if(mode == "lowgrab2"):
        robot_parts[ARM_4].setPosition(1.5)
        robot_parts[ARM_7].setPosition(0)

        grab_counter = grab_counter + 1
        if grab_counter >= 50:
            mode = "lowgrab3"
            grab_counter = 0 

    if(mode == "lowgrab3"):
        robot_parts[ARM_2].setPosition(-0.75)
        robot_parts[ARM_4].setPosition(0.4)
        grab_counter = grab_counter + 1
        if grab_counter >= 50:
            mode = "lowgrab4"
            grab_counter = 0 

    if(mode == "lowgrab4"):
        robot_parts[ARM_2].setPosition(-0.75)
        robot_parts[ARM_4].setPosition(0.4)
        grab_counter = grab_counter + 1
        if grab_counter >= 50:
            mode = "lowgrab5"
            grab_counter = 0 

    if(mode == "lowgrab5"):

        robot_parts[FINGER_LEFT].setPosition(0)#Close fingers (Grab Object)
        robot_parts[FINGER_LEFT].setPosition(0)
        grab_counter = grab_counter + 1
        if grab_counter >= 45:
            mode = "lowgrab6"
            grab_counter = 0

    if(mode == "lowgrab6"):
        robot_parts[ARM_1].setPosition(0.07)

        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "lowgrab7"
            grab_counter = 0

    if(mode == "lowgrab7"):
        robot_parts[ARM_2].setPosition(0.3)
        robot_parts[ARM_3].setPosition(1.5)
        grab_counter = grab_counter + 1
        if grab_counter >= 100:
            mode = "lowgrab8"
            grab_counter = 0

    if(mode == "lowgrab8"):
        robot_parts[ARM_1].setPosition(1.6)
        robot_parts[ARM_6].setPosition(-1.39)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "lowgrab9"
            grab_counter = 0

    if(mode == "lowgrab9"):

        robot_parts[ARM_4].setPosition(2)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "lowgrab10"
            grab_counter = 0

    if(mode == "lowgrab10"):

        robot_parts[ARM_5].setPosition(2)
        grab_counter = grab_counter + 1
        if grab_counter >= 25:
            mode = "lowgrab11"
            grab_counter = 0

    if(mode == "lowgrab11"):

        robot_parts[FINGER_LEFT].setPosition(0.045)
        robot_parts[FINGER_LEFT].setPosition(0.045)
        grab_counter = grab_counter + 1
        if grab_counter >= 50:
            mode = "lowgrab12"
            grab_counter = 0

    if(mode == "lowgrab12"): ##Return arm to the starting position
        for i in range(N_PARTS):
            robot_parts[i].setPosition(float(target_pos[i]))
        grab_counter = grab_counter + 1
        if grab_counter >= 200:
            mode = "manual"
            grab_counter = 0

    # #####################################################
    # #         Distance, Heading, Bearing Error          #
    # #####################################################

    pos_error = math.sqrt((pose_x - goal_pose_x)**2 + (pose_y - goal_pose_y)**2)
    
    # STEP 2.1.1: Bearing Error i.e. How much rotation needed to go toward the goal
    bearing_error = math.atan2((goal_pose_y - pose_y),(goal_pose_x - pose_x)) - pose_theta
    if bearing_error < -3.1415: bearing_error += 6.283
    
    #STEP 2.1.2: Heading Error i.e. How much roation needed to be oriented the same way as the goal
    heading_error = goal_pose_theta - pose_theta
    if heading_error < -3.1415: heading_error += 6.238

    # print("pos: ", pos_error)
    # print("bear: ", bearing_error)
    # print("head: ", heading_error)

    # #####################################################
    # #               Lidar Vision Sections               #
    # #####################################################
    """
    NOTE: Lidar Vision Sections are used to differentiate between the right, middle, and left sides.
    This was used to create a automated right left turing set of functions that guided the robot between
    the rows as it mapped its surrounding/used color blob recognition. 
    
    """

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    # Represented all reported lidar values below a certain threshold. This was done so that the robot could be aware 
    # of how close it was to something that it saw. This did not affect the values of the map and were isolated.

    lidar_close = []

    #################################################################################################################

    for i in range (len(lidar_sensor_readings)):
        if lidar_sensor_readings[i] < 2:
            lidar_close.append(i)

    lidar_left = []

    for k in range (169):
        if lidar_sensor_readings[k] < 2:
            lidar_left.append(k)
            
    lidar_right = []
 
    for j in range (320,len(lidar_sensor_readings)):
        if lidar_sensor_readings[j] < 2:
            lidar_right.append(j)
    
    lidar_middle = []
    left_bound = 180
    right_bound = 300

    for l in range (180,300):
        if lidar_sensor_readings[l] < 2:
            lidar_middle.append(l)
    
    # For testing purposes, see the values of the 4 different arrays

    # print (lidar_close)
    # print ("L",lidar_left)
    # print ("R",lidar_right)
    # print ("M",lidar_middle)

    # ##########################################################
    # #                      Manual Mode                       #
    # ##########################################################
    if mode == 'manual':
        
        # map_func()

        key = keyboard.getKey()

        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('A'):
            mode = ("grab0")
            print("arm mode activated")
        elif key == ord('L'):
            mode = ("lowgrab0")
            print("low arm mode activated")
        elif key == ord('S'):
            
            array = map > 0.5
            array = array * 1
            np.save("map.npy", array, allow_pickle=True, fix_imports=True)

            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")

        else: # slow down
            vL *= 0.75
            vR *= 0.75

    # align method that sets robot in position to use arm. Does not work for all goal objects
    # This code was used for testing and using the arm. Not called in this version of the code.
    elif mode == "align":
        # 1. stop
        # 2. turn 90 degrees towards goal object
        # 3. move towards goal object until differnce = correct distance
        # 4. trigger appropriate arm mode
        if ((lane == 1) or (lane == 3)) and gps.getValues()[1] < goal_positions_world[goal_positions_index][1]:  #object is to the right
            print("right")
            robot_parts[MOTOR_LEFT].setVelocity(MAX_SPEED * 0.2) 
            robot_parts[MOTOR_RIGHT].setVelocity(MAX_SPEED * -0.2)
            where_is_object = "right"
        elif ((lane == 1) or (lane == 3)) and gps.getValues()[1] > goal_positions_world[goal_positions_index][1]:  #object is to the left
            robot_parts[MOTOR_LEFT].setVelocity(MAX_SPEED * -0.2) 
            robot_parts[MOTOR_RIGHT].setVelocity(MAX_SPEED * 0.2)
            print("left")
            where_is_object = "left"
        elif (lane == 2) or (lane == 4) and gps.getValues()[1] > goal_positions_world[goal_positions_index][1]:#object is to the left
            robot_parts[MOTOR_LEFT].setVelocity(MAX_SPEED * 0.2) 
            robot_parts[MOTOR_RIGHT].setVelocity(MAX_SPEED * -0.2)
            where_is_object = "left"
        elif (lane == 2) or (lane == 4) and gps.getValues()[1] < goal_positions_world[goal_positions_index][1]:#object is to the left
            robot_parts[MOTOR_LEFT].setVelocity(MAX_SPEED * -0.2) 
            robot_parts[MOTOR_RIGHT].setVelocity(MAX_SPEED * 0.2)
            where_is_object = "left"

        turn_counter += 1
        print("turncounter:", turn_counter)
        # if turn_counter >= 192.5:
        if turn_counter >= 187:
            turn_counter = 0
            robot_parts[MOTOR_LEFT].setVelocity(0) 
            robot_parts[MOTOR_RIGHT].setVelocity(0)
            mode = "move_forward_a_little"
        # if (lidar_sensor_readings[334] < lidar_sensor_readings[336]) and (lidar_sensor_readings[334] < lidar_sensor_readings[332]):
        #     robot_parts[MOTOR_LEFT].setVelocity(0) 
        #     robot_parts[MOTOR_RIGHT].setVelocity(0)
    

    elif mode == "move_forward_a_little":
        turn_counter += 1
        print(turn_counter)

        robot_parts[MOTOR_LEFT].setVelocity(MAX_SPEED * 0.2) 
        robot_parts[MOTOR_RIGHT].setVelocity(MAX_SPEED * 0.2)

        # if turn_counter >= 115:
        if turn_counter >= 130:
            if goal_positions_world[goal_positions_index][2] >= 1: #if object is on the top shelf
                if abs(gps.getValues()[1] - goal_positions_world[goal_positions_index][1]) > 1: #reach the desired position in relation to the object
                    robot_parts[MOTOR_LEFT].setVelocity(0) 
                    robot_parts[MOTOR_RIGHT].setVelocity(0)
                    turn_counter = 0
                    mode = "grab1"
            elif goal_positions_world[goal_positions_index][2] < 1: #if object is on the second shelf
                if abs(gps.getValues()[1] - goal_positions_world[goal_positions_index][1]) <= 1: #reach the desired position in relation to the object
                    robot_parts[MOTOR_LEFT].setVelocity(0) 
                    robot_parts[MOTOR_RIGHT].setVelocity(0)
                    turn_counter = 0
                    mode = "lowgrab1"

    # ##########################################################
    # #         Map Generation and Color Blob Results          #
    # ##########################################################
    elif mode == 'show':
        map = np.load("map.npy")
        map = np.flip(np.rot90(map))
        print("Map loaded")

        # Show the map
        plt.imshow(map)
        plt.show()

        # Switch mode to manual for object retrival
        mode = 'manual'

        # convert goal_positions to world_positions
        for goal in goal_positions:
            rx = goal[0]
            ry = goal[1]
            t = pose_theta + np.pi/2.
            wx = math.cos(t)*rx - math.sin(t)*ry + pose_x
            wy = math.sin(t)*rx + math.cos(t)*ry + pose_y
            goal_positions_world.append((wx, wy, goal[2]))
        for world in goal_positions_world:
            print(world)

    # starting position: 13, 5.76719
        
    # ##########################################################
    # #            Automated Mapping Turning Logic             #
    # ##########################################################
    """
    NOTE: This will be used to guide the robot through the rows utilizing manipulated lidar values and allow the robot
    to map its surrounding and utilize color blob detection.
    
    Of note is that there are operation function trees for a right and left turn. The right turn was designed specifically
    to handle the crossover between rows without a corner/open space present. 
    
    """
    # # Starts the entire automated process.
    if mode == 'auto_start':

        count += 1
        #############################################################

        if sub_mode == 'start':
            print('start')

            # Draw the robot's current pose on the 360x360 display
            display.setColor(int(0xFF0000))

            display.drawPixel(x_map,y_map)

            # enable mapping during movement
            map_func()
            see_yellow(count)

            # make sure pc is reset
        
            if (len(lidar_left) == 0):
                bottom = False
                sub_mode = 'turn left'

            vL = MAX_SPEED*0.5
            vR = MAX_SPEED*0.5

        # utilizes a fixed vL/vR to turn the robot left, once it recognizes that it is oriented correctly progress to turn left 2 function 
        elif sub_mode == 'turn left':
            print('turnleft')
            vL = MAX_SPEED*0.6
            vR = MAX_SPEED*0.75

            # utilizes the mean value of the collected indexs to gauge when the robot is close to the position expected. 
                # these values were collected and used to get the best results when moving autonomously but can be generalized for
                # a more mixed and unpredicable stopping point
            if (position_counter < 1):
                turn =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 62, 63, 64, 65, 66, 67, 86, 87, 88, 89, 90]
                turnr = [380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476]
            elif (position_counter == 1):
                turn =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
                turnr = [346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484,500,500,500,500,500,500,600]
            
            if (np.mean(lidar_left) >= np.mean(turn) and len(lidar_middle) == 0  and np.mean(lidar_right) >= np.mean(turnr)):
                sub_mode = 'turn left 2'
                # add to position_counter to keep track of number of left turns
                position_counter += 1

        # drives the robot forward and re-enables mapping for strong and clear mapping. Acts as end function after left turns are satisfied
        elif sub_mode == 'turn left 2':
            print('turnleft2')
            
            # count = 50

            # Draw the robot's current pose on the 360x360 display
            display.setColor(int(0xFF0000))
            display.drawPixel(x_map,y_map)
            map_func()
            see_yellow(count)

            # reaches edge of rows, no lidar values read within 2m
            
            if (position_counter == 1):
                vL = MAX_SPEED * 0.500
                vR = MAX_SPEED * 0.500
                if (len(lidar_close) == 0):
                    sub_mode = 'turn right'
            elif (position_counter == 2):
                vL = MAX_SPEED * 0.4990
                vR = MAX_SPEED * 0.501
                end = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 11, 11, 11]
                if (len(lidar_middle) == 0 and len(lidar_right) == 0 and len(lidar_left) >= len(end)):
                    sub_mode = 'turn right 2'
            # end automation once position counter is == 2
            elif (position_counter == 3):
                sub_mode = 'turn right 4'
                vL = MAX_SPEED * 0.5
                vR = MAX_SPEED * 0.5

        elif sub_mode == 'turn end':
            print('turn end')
            map_func()
            see_yellow(count)
            vL = MAX_SPEED*0.4995
            vR = MAX_SPEED*0.5005
            
            if (len(lidar_right) == 0 and len(lidar_middle) == 0 and len(lidar_left) > 0):

                vL = 0
                vR = 0
            
                # save map generated
                array = map > 0.6
                array = array * 1
                np.save("map.npy", array, allow_pickle=True, fix_imports=True)

                mode = 'show'

        # utilizes a fixed vL/vR to turn the robot left, once it recognizes that it is oriented correctly progress to turn left 2 function 
        elif sub_mode == 'turn right':
            print('turnright')
    
            vL = MAX_SPEED*0.75
            vR = MAX_SPEED*0.6
            
            endr = [449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500]
            end = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 104, 105, 106, 107, 108]
            
            if (len(lidar_left) <= len(end) and len(lidar_middle) == 0 and len(lidar_right) >= len(endr)):
                sub_mode = 'turn left 2'
                # add to position_counter to keep track of number of turns
                position_counter += 1

        elif sub_mode == 'turn right 2':
            print('turnright2')
    
            vL = MAX_SPEED*0.18
            vR = MAX_SPEED*-0.17
            endr = [387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,418]

            if (len(lidar_left) == 0 and len(lidar_middle) == 0 and len(lidar_right) >= len(endr)):
                sub_mode = 'turn right 3'
                # add to position_counter to keep track of number of turns
                position_counter += 1

        elif sub_mode == 'turn right 3':
            print('turnright3')

            vL = MAX_SPEED*0.7
            vR = MAX_SPEED*0.699
            endr = [373, 374, 375, 376, 377, 378, 379, 380, 381]
            endm = [187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219]

            if (len(lidar_left) == 0 and len(lidar_middle) >= len(endm) and len(lidar_right) >= len(endr)):
                sub_mode = 'turn right 4'
                position_counter += 1
        
        elif sub_mode == 'turn right 4':
            print('turnright4')

            vL = MAX_SPEED*0.2
            vR = MAX_SPEED* -0.2

            check = True

            endl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 11, 11]
            endr = [180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,11,11]

            if (len(lidar_left) <= len(endl) and len(lidar_middle) <= len(endr) and len(lidar_right) == 0):
                sub_mode = 'turn end'
                vL = MAX_SPEED * 0.5
                vR = MAX_SPEED * 0.5
                check = False
    
    # ##########################################################
    # #                 Dedicated Turning Mode                 #
    # ##########################################################

    # Better turning accuracy, seperate from main automated point
    # following (PFM)

    elif mode == 'turn':

        p1 = 0
        p2 = 1
        
        xdot = p1 * pos_error
        thetadot = p2 * bearing_error + p3 * heading_error

        vL = (xdot - ((thetadot*AXLE_LENGTH)/2))
        vR = (xdot + ((thetadot*AXLE_LENGTH)/2))
        
        if abs(bearing_error) >= 0.2:
            if vL > vR:          
                vL = MAX_SPEED 
                vR = MAX_SPEED * 0.3
            elif vR >= vL:
                vL = MAX_SPEED * 0.3
                vR = MAX_SPEED 
        else:
            mode = 'auto'

    # ##########################################################
    # #                 Point Following Mode                   #
    # ##########################################################

    # Should require points to have been generated from color blob 
    # detection and automated mapping

    elif mode == 'auto': # not manual mode

        if bearing_error <= 0.2:
            p1 = 0.1
            p3 = 0
        else: # not pointed in the rigth direction
            p2 = 0.5 #might not be neccessary
            p1 = 0

        xdot = p1 * pos_error
        thetadot = p2 * bearing_error + p3 * heading_error

        vL = (xdot - ((thetadot*AXLE_LENGTH)/2))
        vR = (xdot + ((thetadot*AXLE_LENGTH)/2))

        # Adjust Speed of the wheels to correct errors (distance,heading,bearing)

        if vL > vR:
            propor_number = vR/vL
            propor_number = propor_number * 0.7
            
            vL = MAX_SPEED  * 0.7
            vR = MAX_SPEED  * propor_number
        elif vR > vL:
            propor_number = vL/vR
            propor_number = propor_number * 0.7
            
            vL = MAX_SPEED * propor_number
            vR = MAX_SPEED * 0.7

        # STEP 2.4: Clamp wheel speeds

        if (vL > MAX_SPEED):
            vL = MAX_SPEED
        if (vL < -1 * MAX_SPEED):
            vL = -1 * MAX_SPEED
        if (vR > MAX_SPEED):
            vR = MAX_SPEED
        if (vR < -1 * MAX_SPEED):
            vR = -1 * MAX_SPEED

        if(pos_error < 0.2 and position_counter != 1):
            mode = 'arm'

        # Robot Stop Edge Case Statement

        if(heading_error <= .02 and pos_error < .2 and position_counter == 1):
            robot_parts[MOTOR_LEFT].setVelocity(0) 
            robot_parts[MOTOR_RIGHT].setVelocity(0)
            break

    elif mode == 'done':
        print('done')
        break

    # #####################################################
    # #                    Odometry                       #
    # #####################################################

    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))
    print("vL: %f vR: %f" % (vL, vR))
    print("Goal Objs:", len(goal_positions))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
