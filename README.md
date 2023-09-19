**FINAL PROJECT CONTROLLER** 

Used to navigate a Tiago Steel robot through a grocery store simulation.

**REQUIREMENTS TO RUN THIS SIMULATION**
- Computer with modern CPU and Graphics Intergrated or Dedicated (2018 or newer AMD Ryzen 3 APU or Intel Core
  i5 10600 or newer)
- WEBOTS version 2022b (may not run correctly on a newer instance.) Link to download:
  https://github.com/cyberbotics/webots/releases/tag/R2022b (or google webots old installs)
- Python Version 8.0 or newer is required. 

**A link to a demo video:** https://drive.google.com/file/d/1DEh72CK1_E5N0jJSyjkVC0tQLvZea-Ee/view?usp=drive_link

**This was a team project developed for an Intro to Robotics class and contains some notable feature
functions:**

1. Computer Vision Color Detection give the robot the ability to use its camera to identify target objects
   required for retrival with the arm. Target objects are colored green/yellow while mics. items are
   colored differently to prove the color mapping system works.
2. Auto mapping was implemented for the robot and allows it to navigate its surrounding, using a lidar
   system to detect the ends of each row and turn accordingly using a combination of kinematics and sensor
   detection. Lidar is also used to plot mapping points on an appropriatly scaled map of the simulated world
   and was designed for automated traversal. Unfortunatly that feature was cut following time constraints and
   was never fully implemented in this controller.
3. Manual arm control was added so that a pilot could manually control the arm of the Tiago Steel robot during
   the discovery and retrival of a target item. A combination of the robot's auto mapping working with color
   detection marked all the target items in the sim. When pilot was given control after the sequence was complete,
   the robot was stop and align itself accordingly when it reach a previously marked target. This greatly helped
   with the accuracy of the arm operation by the pilot and reduced time inbetween target objects.   
