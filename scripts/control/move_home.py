#!/usr/bin/env python3
"""Move UR5e to upright home position via MoveIt2."""
import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import Constraints, JointConstraint

HOME = {
    'arm0_shoulder_pan_joint':   0.0,
    'arm0_shoulder_lift_joint': -1.5708,
    'arm0_elbow_joint':          1.5708,
    'arm0_wrist_1_joint':       -1.5708,
    'arm0_wrist_2_joint':        0.0,
    'arm0_wrist_3_joint':        0.0,
}

class MoveHome(Node):
    def __init__(self):
        super().__init__('move_home')
        self._client = ActionClient(self, MoveGroup, '/move_group')
        self.get_logger().info('Waiting for MoveGroup...')
        self._client.wait_for_server()
        self.get_logger().info('Connected! Sending home position...')
        self._send_goal()

    def _send_goal(self):
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = 'ur_manipulator'
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 10.0
        req.max_velocity_scaling_factor     = 0.3
        req.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        for joint_name, position in HOME.items():
            jc                 = JointConstraint()
            jc.joint_name      = joint_name
            jc.position        = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)
        self._client.send_goal_async(goal).add_done_callback(self._goal_cb)

    def _goal_cb(self, future):
        future.result().get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info('✅ Home position reached!')
        else:
            self.get_logger().error(f'❌ Failed with status: {status}')
        raise SystemExit

def main():
    rclpy.init()
    try:
        rclpy.spin(MoveHome())
    except SystemExit:
        pass
    rclpy.shutdown()

main()
