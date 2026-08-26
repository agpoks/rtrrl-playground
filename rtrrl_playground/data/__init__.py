"""Readers for real recordings, so the agents here can be trained on real data.

Everything else in this repo is simulated. This is the seam where it stops
being: :mod:`rtrrl_playground.data.rosbag` turns a ROS 2 recording of an actual
car -- lidar, drive commands, odometry -- into exactly the two things the
offline half of the pipeline needs, a stream of ``(observation, expert action)``
pairs and the track the recording was made on.

That is the setting from Lemmel, Resch, Farsang, Hasani, Rus & Grosu
(arXiv:2602.02236): clone a controller from demonstrations offline, then
fine-tune it online with RTRRL while the car drives. The cloning half runs on
whatever you have already recorded; only the second half needs the car.
"""

from rtrrl_playground.data.rosbag import (
    BagDemonstrations, bag_summary, load_map, load_odometry, pick_pose_topic,
    track_from_bag,
)

__all__ = ["BagDemonstrations", "bag_summary", "load_map", "load_odometry",
           "pick_pose_topic", "track_from_bag"]
