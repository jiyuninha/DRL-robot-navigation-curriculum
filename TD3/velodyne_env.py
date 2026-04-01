import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from gazebo_msgs.srv import GetWorldProperties

# -------------------------------------------------------------------
def wait_for_gazebo(timeout=30.0):
    """/gazebo/get_world_properties 서비스가 준비될 때까지 대기."""
    start = time.time()
    while True:
        try:
            rospy.wait_for_service('/gazebo/get_world_properties', timeout=1.0)
            return True
        except Exception:
            if time.time() - start > timeout:
                rospy.logerr("Timed out waiting for /gazebo/get_world_properties")
                return False
            time.sleep(0.1)

def model_exists(model_name, timeout=1.0):
    """Gazebo world에 model_name이 존재하는지 확인."""
    try:
        rospy.wait_for_service('/gazebo/get_world_properties', timeout=timeout)
        gwp = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        resp = gwp()
        return model_name in resp.model_names
    except Exception as e:
        rospy.logwarn("model_exists: get_world_properties failed: %s" % e)
        return False
# -------------------------------------------------------------------


GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

# =========================
# Footprint / clearance params
# =========================
ROBOT_RADIUS = 0.22
SAFETY_MARGIN = 0.10
CLEARANCE_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN

GOAL_CLEARANCE_ANGLE_SAMPLES = 36
GOAL_CLEARANCE_RADIAL_SAMPLES = 6

BOX_CLEARANCE_MARGIN = 0.20
SPAWN_CLEARANCE_RADIUS = 0.85
GOAL_STRICT_RADIUS = 1.00
GOAL_RELAXED_RADIUS = 0.80
GOAL_INNER_MARGIN = 0.18
PATH_STRICT_MARGIN = 0.30
PATH_RELAXED_MARGIN = 0.25


def is_in_inflated_rect(x, y, xmin, xmax, ymin, ymax, margin):
    return (xmin - margin) <= x <= (xmax + margin) and \
           (ymin - margin) <= y <= (ymax + margin)


def check_pos(x, y, margin=CLEARANCE_RADIUS):
    """
    고정 Gazebo world 기준 point validity 검사.
    margin만큼 obstacle inflation을 적용한다.

    주의:
    - 회전된 wall들은 보수적으로 axis-aligned bounding box(AABB)로 막음
    - table은 다리만이 아니라 tabletop footprint 전체를 보수적으로 막음
      (goal이 table 아래로 생성되는 것까지 막기 위함)
    - cardboard_box_*와 small_crate_1도 현재 고정 world의 장애물로 포함
    """
    # -------------------------------------------------
    # 1) outer boundary
    # current code / sampling 범위와 맞춰 보수적으로 유지
    # -------------------------------------------------
    if x > 4.5 - margin or x < -4.5 + margin or y > 4.5 - margin or y < -4.5 + margin:
        return False

    # -------------------------------------------------
    # 2) fixed obstacles from world file
    # format: (xmin, xmax, ymin, ymax)
    # -------------------------------------------------
    fixed_rects = [
        # ===== interior walls around (-2.95, 2.6) =====
        (-4.20, -1.70,  2.58,  2.73),   # Wall_11
        (-3.03, -2.88,  1.19,  3.94),   # Wall_13

        # ===== top-right room / enclosure =====
        ( 2.89,  4.16,  3.75,  3.90),   # Wall_15
        ( 2.93,  4.18,  2.94,  3.09),   # Wall_17
        ( 3.99,  4.19,  2.94,  3.90),   # Wall_16 (rotated, conservative AABB)
        ( 2.88,  3.08,  2.94,  3.90),   # Wall_18 (rotated, conservative AABB)

        # ===== bottom-left angled enclosure =====
        (-4.22, -3.18, -4.21, -2.38),   # Wall_24 (rotated, conservative AABB)
        (-3.38, -2.16, -4.21, -2.38),   # Wall_25 (rotated, conservative AABB)
        (-4.19, -2.19, -4.16, -4.01),   # Wall_26

        # ===== right-bottom corridor / corner =====
        ( 2.31,  4.06, -3.36, -3.21),   # Wall_28
        ( 3.93,  4.08, -3.34, -0.84),   # Wall_29

        # ===== bookshelf =====
        # bookshelf pose: (4.78412, -3.73836), overall conservative footprint
        ( 4.33,  5.24, -4.14, -3.72),

        # ===== table =====
        # table pose: (-4.34248, -0.532954), tabletop footprint 전체를 보수적으로 차단
        (-5.10, -3.59, -0.93, -0.13),

        # ===== cardboard_box_* (fixed world positions) =====
        # size = 0.5 x 0.4
        ( 3.72,  4.22,  0.78,  1.18),   # cardboard_box_0
        (-0.25,  0.26, -4.17, -3.76),   # cardboard_box_1
        (-0.27,  0.24,  3.79,  4.19),   # cardboard_box_2
        ( 1.55,  2.05,  1.00,  1.40),   # cardboard_box_3
        (-1.85, -1.35,  0.80,  1.20),   # cardboard_box_4
        ( 0.65,  1.15, -1.70, -1.30),   # cardboard_box_5
        (-2.45, -1.95, -1.60, -1.20),   # cardboard_box_6

        # ===== small crate =====
        # size = 0.6 x 0.5
        ( 0.70,  1.30,  2.55,  3.05),   # small_crate_1
    ]

    for xmin, xmax, ymin, ymax in fixed_rects:
        if is_in_inflated_rect(x, y, xmin, xmax, ymin, ymax, margin):
            return False

    # -------------------------------------------------
    # 3) optional: fire hydrant near top-left corner
    # 거의 boundary에 붙어 있어서 boundary 검사로 대부분 걸러지지만,
    # 혹시 몰라 보수적으로 원형 금지영역 추가
    # -------------------------------------------------
    fire_hydrant_x = -4.57173
    fire_hydrant_y =  4.55086
    fire_hydrant_radius = 0.20

    if (x - fire_hydrant_x) ** 2 + (y - fire_hydrant_y) ** 2 <= (fire_hydrant_radius + margin) ** 2:
        return False

    return True


def check_goal_clearance(goal_x, goal_y, clearance=CLEARANCE_RADIUS, grid_step=0.05, inner_margin=GOAL_INNER_MARGIN):
    """
    goal 중심 반경 clearance 내부가 실제로 충분히 비어 있는지
    격자 기반으로 더 엄격하게 검사한다.
    """
    if not check_pos(goal_x, goal_y, margin=inner_margin):
        return False

    x_vals = np.arange(goal_x - clearance, goal_x + clearance + grid_step, grid_step)
    y_vals = np.arange(goal_y - clearance, goal_y + clearance + grid_step, grid_step)

    for x in x_vals:
        for y in y_vals:
            if (x - goal_x) ** 2 + (y - goal_y) ** 2 <= clearance ** 2:
                if not check_pos(x, y, margin=inner_margin):
                    return False

    return True


def is_pose_valid(x, y, clearance=CLEARANCE_RADIUS, inner_margin=GOAL_INNER_MARGIN):
    """
    spawn / goal 공통 validity 검사:
    1) inflated obstacle 기준 빠른 검사
    2) 주변 clearance 격자 검사
    """
    if not check_pos(x, y, margin=clearance):
        return False

    if not check_goal_clearance(x, y, clearance=clearance, inner_margin=inner_margin):
        return False

    return True

def is_path_clear(start_x, start_y, goal_x, goal_y, margin=0.25, step=0.10):
    """
    현재 로봇 위치에서 goal까지의 직선 경로를 따라
    최소한의 접근 가능성이 있는지 검사한다.
    완전한 planner는 아니지만, 너무 좁은 곳 / 막힌 곳을 많이 걸러준다.
    """
    dx = goal_x - start_x
    dy = goal_y - start_y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 1e-6:
        return False

    num_steps = max(2, int(dist / step))

    for i in range(1, num_steps + 1):
        t = float(i) / float(num_steps)
        x = start_x + t * dx
        y = start_y + t * dy

        if not check_pos(x, y, margin=margin):
            return False

    return True


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        if not wait_for_gazebo(timeout=30.0):
            raise RuntimeError("Gazebo didn't start or /gazebo/get_world_properties unavailable")

        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

        try:
            rospy.wait_for_message("/r1/odom", Odometry, timeout=5.0)
            rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=5.0)
        except Exception:
            rospy.logwarn("Timeout waiting for initial odom/velodyne messages")

    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                if mag1 == 0 or mag2 == 0:
                    continue
                beta = math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def step(self, action):
        target = False

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        start_o = time.time()
        while self.last_odom is None and time.time() - start_o < 1.0:
            time.sleep(0.005)
        if self.last_odom is None:
            rospy.logwarn_throttle(5.0, "No odom message yet; using zero odom temporarily")
            empty_odom = Odometry()
            empty_odom.pose.pose.position.x = 0.0
            empty_odom.pose.pose.position.y = 0.0
            empty_odom.pose.pose.position.z = 0.0
            empty_odom.pose.pose.orientation.w = 1.0
            self.last_odom = empty_odom

        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = 1.0
        if mag1 == 0:
            beta = 0.0
        else:
            beta = math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = -beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        expected_models = ["r1"] + [f"cardboard_box_{i}" for i in range(7)]
        start = time.time()
        timeout = 8.0
        all_ok = False
        while time.time() - start < timeout:
            all_ok = True
            for m in expected_models:
                if not model_exists(m, timeout=0.4):
                    all_ok = False
                    break
            if all_ok:
                break
            time.sleep(0.1)
        if not all_ok:
            rospy.logwarn("Not all expected models present after reset; some reset calls may fail")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0.0
        y = 0.0
        position_ok = False
        max_trials = 3000

        for _ in range(max_trials):
            x = np.random.uniform(-4.2, 4.2)
            y = np.random.uniform(-4.2, 4.2)
            if is_pose_valid(x, y, clearance=SPAWN_CLEARANCE_RADIUS, inner_margin=0.15):
                position_ok = True
                break

        if not position_ok:
            raise RuntimeError("Failed to find robot spawn with sufficient surrounding clearance.")

        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        self.change_goal()
        # self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        try:
            rospy.wait_for_message("/velodyne_points", PointCloud2, timeout=2.0)
        except Exception:
            rospy.logwarn("No velodyne_points received within timeout after reset")

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = 1.0
        if mag1 == 0:
            beta = 0.0
        else:
            beta = math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = -beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        sample_min_x = max(-3.8, self.odom_x + self.lower)
        sample_max_x = min(3.8, self.odom_x + self.upper)
        sample_min_y = max(-3.8, self.odom_y + self.lower)
        sample_max_y = min(3.8, self.odom_y + self.upper)

        min_goal_dist_from_robot = 1.8
        max_goal_dist_from_robot = 5.5

        # 최근 goal과 너무 가까운 위치 반복 방지용
        # self.recent_goals가 없으면 여기서 생성
        if not hasattr(self, "recent_goals"):
            self.recent_goals = []

        recent_goal_min_dist = 1.5
        max_recent_goals = 8

        max_trials = 4000

        def far_from_recent_goals(x, y):
            for gx, gy in self.recent_goals:
                if np.linalg.norm([x - gx, y - gy]) < recent_goal_min_dist:
                    return False
            return True

        def candidate_score(x, y, distance_to_robot, target_dist):
            """
            점수가 작을수록 더 좋은 후보.
            - 목표 거리(target_dist)에 가까울수록 좋음
            - 최근 goal들과 멀수록 좋음
            - map 중앙/특정 corridor에만 몰리지 않도록 약간의 랜덤성 부여
            """
            dist_term = abs(distance_to_robot - target_dist)

            if len(self.recent_goals) > 0:
                min_recent_dist = min(np.linalg.norm([x - gx, y - gy]) for gx, gy in self.recent_goals)
            else:
                min_recent_dist = 3.0

            # 최근 goal과 멀수록 유리하게
            diversity_bonus = -0.35 * min(min_recent_dist, 3.0)

            # 완전 deterministic하게 best 하나만 고르지 않도록 작은 랜덤성 추가
            random_jitter = random.uniform(0.0, 0.25)

            return dist_term + diversity_bonus + random_jitter

        # strict -> relaxed -> fallback 순서
        phase_settings = [
            ("strict", GOAL_STRICT_RADIUS, PATH_STRICT_MARGIN, 0.10, 2.8, 4.2),
            ("relaxed", GOAL_RELAXED_RADIUS, PATH_RELAXED_MARGIN, 0.08, 2.3, 4.5),
            ("fallback", 0.55, 0.18, 0.10, 2.0, 4.8),
        ]

        for label, radius, path_margin, path_step, target_dist_min, target_dist_max in phase_settings:
            valid_candidates = []
            target_dist = random.uniform(target_dist_min, target_dist_max)

            for _ in range(max_trials):
                cand_x = random.uniform(sample_min_x, sample_max_x)
                cand_y = random.uniform(sample_min_y, sample_max_y)

                distance_to_robot = np.linalg.norm([cand_x - self.odom_x, cand_y - self.odom_y])

                if distance_to_robot < min_goal_dist_from_robot:
                    continue
                if distance_to_robot > max_goal_dist_from_robot:
                    continue

                # 최근 goal과 너무 가까운 위치는 우선 제외
                if not far_from_recent_goals(cand_x, cand_y):
                    continue

                if not is_pose_valid(
                    cand_x,
                    cand_y,
                    clearance=radius,
                    inner_margin=GOAL_INNER_MARGIN if label != "fallback" else 0.08,
                ):
                    continue

                # fallback에서는 path 조건을 약간 완화
                if label != "fallback":
                    if not is_path_clear(
                        self.odom_x,
                        self.odom_y,
                        cand_x,
                        cand_y,
                        margin=path_margin,
                        step=path_step,
                    ):
                        continue
                else:
                    # fallback은 path_clear를 완전히 버리진 않되 더 느슨하게
                    if not is_path_clear(
                        self.odom_x,
                        self.odom_y,
                        cand_x,
                        cand_y,
                        margin=0.14,
                        step=0.12,
                    ):
                        continue

                score = candidate_score(cand_x, cand_y, distance_to_robot, target_dist)
                valid_candidates.append((score, cand_x, cand_y))

            if len(valid_candidates) > 0:
                valid_candidates.sort(key=lambda item: item[0])

                # 상위 후보들 중 하나를 랜덤하게 선택해서
                # 계속 같은 위치만 반복되지 않도록 함
                top_k = min(15, len(valid_candidates))
                chosen = random.choice(valid_candidates[:top_k])

                self.goal_x = chosen[1]
                self.goal_y = chosen[2]

                self.recent_goals.append((self.goal_x, self.goal_y))
                if len(self.recent_goals) > max_recent_goals:
                    self.recent_goals.pop(0)

                return

            rospy.logwarn("Failed to find %s valid goal; trying next phase." % label)

        # 정말 못 찾는 경우: 이전 goal 유지
        rospy.logwarn("Failed to find valid goal; keeping previous goal.")
        return

    def random_box(self):
        for i in range(7):
            name = "cardboard_box_" + str(i)
            start_re = time.time()
            found = False
            while time.time() - start_re < 1.0:
                if model_exists(name, timeout=0.4):
                    found = True
                    break
                time.sleep(0.05)
            if not found:
                rospy.logwarn("%s not present in Gazebo; skipping set_model_state for this model" % name)
                continue

            x = 0.0
            y = 0.0
            box_ok = False
            max_trials = 500

            for _ in range(max_trials):
                x = np.random.uniform(-4.5, 4.5)
                y = np.random.uniform(-4.5, 4.5)

                box_ok = check_pos(x, y, margin=BOX_CLEARANCE_MARGIN)
                if not box_ok:
                    continue

                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
                    continue

                break

            if not box_ok:
                rospy.logwarn("Failed to place %s with clearance; skipping placement update." % name)
                continue

            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2