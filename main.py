import cv2 as cv
import numpy as np
import sys
import math
import json

"""
*****************************************************************
                        The Mission
*****************************************************************
- This program will identify drops by position of each drop.
- The output should be: a dictionary in this format: 
    res = {"0" : [start_frame,[(x0,y0),(x1,y1)....]], "1" : [start_frame,[(x0,y0),....]]}
- The keys of the dictionary will be a unique if dor each drop.
- The value for each key will be a list that has the first frame the drop showed in as the first entry,
  and a second entry will be a list of tuples that has the drop location by order of the frames, from the 
  first the drop appears till it gone. 
"""

"""
*****************************************************************
                        Global Variables
*****************************************************************
"""

frames = []
res = {}
counter = 0
merged = False
Remember = 0
count_speeds = True
points_frame_i = []
new_points = []

"""
*****************************************************************
                    Read Frames
*****************************************************************
- Read all the frames using openCV, and put it in array of frames
"""

def readFrames():
    for i in range(146):
        frame_path = f"video_test_students/frame{i}.jpg"
        frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        frames.append(frame)

"""
*****************************************************************
                    Find all the drops
*****************************************************************
- This function will receive a frame and find all the locations (x,y) of the drops.
- Once we found a drop that we never saw, we need to add it to the dictionary, and 
  save the frame number the drop first appeared
"""

def findDrops(frame, frame_num):
    global counter
    global drops_num
    _, thresh = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    centers = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        centers.append(center)
        cv.rectangle(frame, (x, y), (x + w, y + h), 255, 1)
    drops_num = len(centers)
    return centers

"""
*****************************************************************
                    Find distances 
*****************************************************************
- This function will receive a drop and will calculate the distance between her and all
  other drops that *above* her in the last frame. we dont need to check the points below the drop because all 
  the drops moving down as time pass.
"""

def find_distances(center, last_frame_centers):
    distances = []
    for last_center in last_frame_centers:
        distances.append([last_center, math.dist(center, last_center)])
    return distances

"""
*****************************************************************
                    Get Key
*****************************************************************
- For merge, we need to iterate all res in order to find the points that belong to the collided points
"""

def get_key(val):
    for key, value in res.items():
        for pt in value[1]:
            if pt == val:
                return key

"""
*****************************************************************
                    Get Key
*****************************************************************
- For merge, we need to iterate all res in order to find the points that belong to the collided points
"""

def get_id(val):
    last_value = {}
    for key, value in res.items():
        last_value[key] = [value[1][-1]]
    for key, value in last_value.items():
        for pt in value:
            if pt == val:
                return key
"""
*****************************************************************
                    Calculate Distance
*****************************************************************
- Calculate distance between 2 points
"""

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

"""
*****************************************************************
                    Find Min Point
*****************************************************************
- After we find the distances between the drop in the current frame, and the drops in the last frame,
  we return the minimum point from last frame to the drop we currently track
"""

def find_min_point(last_frame_distances):
    min_point = distances[0]
    for k in range(len(last_frame_distances)):
        if (last_frame_distances[k][1] <= min_point[1]):
            min_point = distances[k]
    return min_point

"""
*****************************************************************
                    Find Closest Points
*****************************************************************
- In case of merge, we need to find the closest points in the frame after the collision
- This function returns just that
"""

def find_closest_points(points):
    min_distance = float('inf')
    closest_points = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i], points[j])
            if distance < min_distance:
                min_distance = distance
                closest_points = (points[i], points[j])
    return closest_points

"""
*****************************************************************
                    Is Merged?
*****************************************************************
- Finds the 2 closest points in the previous frame(before the collision) and calculate the distance between them.
- If the distance is small enough, return True
"""

def isMerged(points_frame_i):
    points = find_closest_points(points_frame_i)
    dist = calculate_distance(points[0], points[1])
    if (dist < 11):
        return True
    return False

"""
*****************************************************************
                        Main 
*****************************************************************
"""

readFrames()
first_frame = 29
for i in range(first_frame, 145):
    frame = frames[i + 1]
    points_frame_i_plus = findDrops(frame, i + 1)
    num_substruct_points = len(points_frame_i_plus) - len(points_frame_i)
    num_points_frame_i_plus = len(points_frame_i_plus)
    num_points_frame_i = len(points_frame_i)
    # If in the current frame we dont have drops, save the drops as the old drops and continue
    if (num_points_frame_i_plus == 0):
        points_frame_i = points_frame_i_plus.copy()
        continue
    # If there are new drops, it can be 2 things: 2 points collided and in this frame they separate or there are new drops
    if (num_points_frame_i_plus > num_points_frame_i):
        # Option 1: There are new drops
        if (not merged):
            points_frame_i_plus.sort(key=lambda x: x[1])
            for j in range(num_substruct_points):
                new_points.append(points_frame_i_plus[j])
                res[str(counter)] = [i + 1, [points_frame_i_plus[j]]]
                counter += 1
        # Option 2: 2 points collided and in this frame they separate
        else:
            Remember = 1
    # Only when There are less or equal amount of drops we count the speed od each drop(To separate between close drops)
    if (num_substruct_points > 0):
        count_speeds = False
    else:
        count_speeds = True
    # One cycle after Two points are merged we turn it to False, there is no collision anymore
    merged = False
    # If the drops are merged\ if one drop is out of the picture
    if (num_points_frame_i_plus < num_points_frame_i):
        # If to check whether there is a merge
        if (isMerged(points_frame_i)):
            merged = True
            closest_points = find_closest_points(points_frame_i)
            point_id1, point_id2 = get_key(closest_points[0]), get_key(closest_points[1])

    # tracking the drops that exists
    if (num_points_frame_i_plus > 0):
        # substract the new drops from the array of all the points in the frame. we don't need to track a new drop.
        points_to_track = []
        for point in points_frame_i_plus:
            if (point not in new_points):
                points_to_track.append(point)
        for point in points_to_track:
            # Find distance from every point to any point in the last frame
            distances = find_distances(point, points_frame_i)
            # Find the point(from previous frame) that is the nearest to the point
            min_point = find_min_point(distances)
            # Get the key value in the result dictionary that match the nearest point.
            # this is the point in res that we want to put the point we track
            point_id = get_id(min_point[0])
            # **Only when there is a Merge**
            if (merged):
                if (point_id == point_id1):
                    res[point_id2][1].append(point)
                    intersection_point = point
                elif (point_id == point_id2):
                    res[point_id1][1].append(point)
                    intersection_point = point
            # **Only when the current frame is the one after Merge
            # **We run 2 times, this is why the if statment there
            if (0 < Remember < 3):
                # Find 2 closest points, they are the ones that about to collide
                closest_points_after = find_closest_points(points_frame_i_plus)
                # find direction after the collision. positive : right , negative: left
                dir = point[0] - intersection_point[0]
                if (dir > 0):
                    min = min(x for x in closest_points)
                    point_id = get_key(min)
                else:
                    max = max(x for x in closest_points)
                    point_id = get_key(max)
                Remember += 1
            # Put the point in the final answer
            res[point_id][1].append(point)
            # draw the id near the point
            cv.putText(frame, point_id, point,cv.FONT_HERSHEY_SIMPLEX, 0.9, 255, 1)
            # If we want to count the speeds of the points, we calculate the point that match to the current speed,
            # and since we already find place for him in res, we remove it from last frame points. we dont need it anymore.
            if (count_speeds):
                prev_point = res[point_id][1][-2]
                points_frame_i.remove(prev_point)

        # Show drops
        cv.imshow('Contours', frame)
        cv.waitKey(0)

    print(res)
    # Save the points that now new, as old, in order to advance to next frame
    points_frame_i = points_frame_i_plus.copy()

# Export json file
with open("sample.json", "w") as outfile:
    json.dump(res, outfile)
