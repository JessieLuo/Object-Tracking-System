# tracking/builder.py

from tracking.tracker import Tracker

from tracking.motion.model import KalmanMotion

from tracking.appearance.osnet import OSNetReID
from tracking.appearance.memory import AppearanceStore
from tracking.appearance.model import ScheduledAppearance

from tracking.association.associate import (
    MotionAppearanceAssociation,
)


def build_tracker(args):

    # =====================================================
    # reid
    # =====================================================

    reid = OSNetReID(
        model_name=args.reid_model_name,
        weights_path=args.reid_weights,
        min_h=80,
    )

    # =====================================================
    # motion
    # =====================================================

    motion = KalmanMotion()

    # =====================================================
    # appearance
    # =====================================================

    appearance = ScheduledAppearance(
        reid=reid,

        memory=AppearanceStore(
            bank_size=30,
            ema_alpha=0.9,
        ),

        interval=5,

        max_per_frame=4,
    )

    # =====================================================
    # association
    # =====================================================

    association = MotionAppearanceAssociation(
        active_iou_thresh=0.7,
        lost_iou_thresh=0.9,
        motion_thresh=9.4877,
    )

    # =====================================================
    # tracker
    # =====================================================

    return Tracker(
        association=association,

        motion=motion,

        appearance=appearance,

        max_missed=60,

        min_hits=2,

        soft_lost=10,
    )