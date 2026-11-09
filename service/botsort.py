"""Independent BoT-SORT: XYWH Kalman, two-stage IoU matching, and camera motion.

Based on the MIT-licensed algorithm by Nir Aharon et al. (2022),
https://github.com/NirAharon/BoT-SORT. No detector/framework dependency.
ReID remains disabled, as in the serving contract. IDs belong to one stream.
"""

from __future__ import annotations

import logging
from enum import IntEnum

import cv2
import lap
import numpy as np

from service._vendor.botsort.kalman_filter import KalmanFilter

LOGGER = logging.getLogger(__name__)


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    def __init__(self, xywh, score, cls, index):
        self._xywh = np.asarray(xywh, dtype=np.float32)
        self.score, self.cls, self.index = float(score), int(cls), int(index)
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.mean = self.covariance = None
        self.frame_id = self.start_frame = 0

    @property
    def xyxy(self):
        xywh = self._xywh if self.mean is None else self.mean[:4]
        return np.r_[xywh[:2] - xywh[2:] / 2, xywh[:2] + xywh[2:] / 2]

    @property
    def end_frame(self):
        return self.frame_id

    def activate(self, kalman, frame_id, track_id):
        self.mean, self.covariance = kalman.initiate(self._xywh)
        self.frame_id = self.start_frame = frame_id
        self.track_id, self.state = track_id, TrackState.Tracked
        self.is_activated = frame_id == 1

    def update(self, other, frame_id, kalman):
        self.mean, self.covariance = kalman.update(self.mean, self.covariance, other._xywh)
        self.frame_id, self.state, self.is_activated = frame_id, TrackState.Tracked, True
        self.score, self.cls, self.index = other.score, other.cls, other.index


def iou_distance(first, second):
    if not first or not second:
        return np.empty((len(first), len(second)), dtype=np.float32)
    a, b = np.asarray([t.xyxy for t in first]), np.asarray([t.xyxy for t in second])
    overlap = np.maximum(
        0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    ).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return 1 - overlap / np.maximum(area_a[:, None] + area_b[None, :] - overlap, 1e-7)


def associate(first, second, threshold, fuse_score=False):
    costs = iou_distance(first, second)
    if not first or not second:
        return [], list(range(len(first))), list(range(len(second)))
    if fuse_score:
        costs = 1 - (1 - costs) * np.asarray([t.score for t in second])[None, :]
    # A face track must never inherit a plate's whitelist approval or identity.
    costs[
        np.asarray([t.cls for t in first])[:, None] != np.asarray([t.cls for t in second])[None, :]
    ] = 1
    _, rows, cols = lap.lapjv(costs, extend_cost=True, cost_limit=threshold)
    return (
        [(i, int(j)) for i, j in enumerate(rows) if j >= 0],
        list(np.where(rows < 0)[0]),
        list(np.where(cols < 0)[0]),
    )


def multi_gmc(tracks, warp):
    transform = np.kron(np.eye(4), warp[:, :2])
    for track in tracks:
        track.mean = transform @ track.mean
        track.mean[:2] += warp[:, 2]
        track.covariance = transform @ track.covariance @ transform.T


class GMC:
    """OpenCV camera motion, with safe reset on frame-size/feature loss."""

    def __init__(self, method="sparseOptFlow", downscale=2):
        self.method = "none" if method in (None, "None", "none") else method
        if self.method not in ("none", "sparseOptFlow", "orb", "sift", "ecc"):
            raise ValueError(f"unknown GMC method: {method}")
        self.downscale = downscale
        self.prev_frame = self.prev_points = self.prev_descriptors = None
        if self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif self.method == "sift":
            self.detector = cv2.SIFT_create(
                nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20
            )

    def apply(self, image, detections=None):
        identity = np.eye(2, 3, dtype=np.float64)
        if self.method == "none":
            return identity
        h, w = image.shape[:2]
        frame = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            (max(1, w // self.downscale), max(1, h // self.downscale)),
        )
        if self.prev_frame is not None and self.prev_frame.shape != frame.shape:
            self.prev_frame = self.prev_points = self.prev_descriptors = None
        before = self.prev_frame
        self.prev_frame = frame
        if self.method == "ecc":
            if before is None:
                return identity
            _, warp = cv2.findTransformECC(
                before,
                frame,
                identity.astype(np.float32),
                cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            )
            warp[:, 2] *= self.downscale
            return warp
        if self.method == "sparseOptFlow":
            old = self.prev_points
            self.prev_points = cv2.goodFeaturesToTrack(frame, 1000, 0.01, 1, blockSize=3)
            if before is None or old is None or len(old) < 5:
                return identity
            new, status, _ = cv2.calcOpticalFlowPyrLK(before, frame, old, None)
            if new is None or status is None:
                return identity
            valid = status.ravel() == 1
            old, new = old[valid], new[valid]
        else:
            mask = np.zeros_like(frame)
            fh, fw = frame.shape
            mask[int(0.02 * fh) : int(0.98 * fh), int(0.02 * fw) : int(0.98 * fw)] = 255
            if detections is not None:
                for box in detections:
                    x1, y1, x2, y2 = np.maximum(0, np.asarray(box[:4]) / self.downscale).astype(int)
                    mask[y1:y2, x1:x2] = 0
            points, descriptors = self.detector.detectAndCompute(frame, mask)
            old_points, old_descriptors = self.prev_points, self.prev_descriptors
            self.prev_points, self.prev_descriptors = points, descriptors
            if old_descriptors is None or descriptors is None or len(descriptors) < 2:
                return identity
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING if self.method == "orb" else cv2.NORM_L2)
            pairs = matcher.knnMatch(old_descriptors, descriptors, k=2)
            matches = [
                a
                for pair in pairs
                if len(pair) == 2
                for a, b in [pair]
                if a.distance < 0.9 * b.distance
            ]
            old = np.asarray([old_points[m.queryIdx].pt for m in matches], dtype=np.float32)
            new = np.asarray([points[m.trainIdx].pt for m in matches], dtype=np.float32)
        if len(old) < 5:
            return identity
        warp, _ = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC)
        if warp is None or not np.isfinite(warp).all():
            return identity
        warp[:, 2] *= self.downscale
        return warp


class BOTSORT:
    def __init__(self, args):
        if args.with_reid:
            raise ValueError("serving BoT-SORT does not enable ReID")
        self.args = args
        self.frame_id = self._next_id = 0
        self.max_time_lost = int(args.track_buffer)
        self.tracked_stracks, self.lost_stracks, self.removed_stracks = [], [], []
        self.kalman_filter = KalmanFilter()
        self.gmc = GMC(args.gmc_method)
        self._prepared_warp = None

    def reset(self):
        self.__init__(self.args)

    def update(self, boxes, image):
        self.frame_id += 1
        high, low = [], []
        for i, (xywh, score, cls) in enumerate(zip(boxes.xywh, boxes.conf, boxes.cls, strict=True)):
            track = Track(xywh, score, cls, i)
            if score >= self.args.track_high_thresh:
                high.append(track)
            elif score > self.args.track_low_thresh:
                low.append(track)
        active = [t for t in self.tracked_stracks if t.is_activated]
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        pool = list({t.track_id: t for t in active + self.lost_stracks}.values())
        if pool:
            means = np.asarray([t.mean.copy() for t in pool])
            covs = np.asarray([t.covariance for t in pool])
            for i, track in enumerate(pool):
                if track.state != TrackState.Tracked:
                    means[i, 6:8] = 0
            means, covs = self.kalman_filter.multi_predict(means, covs)
            for track, mean, cov in zip(pool, means, covs, strict=True):
                track.mean, track.covariance = mean, cov
        warp, self._prepared_warp = self._prepared_warp, None
        try:
            if isinstance(warp, Exception):
                raise warp
            if warp is None:
                warp = self.gmc.apply(image, np.asarray([t.xyxy for t in high]))
        except Exception as error:
            LOGGER.warning("GMC failed, falling back to identity: %s", error)
            warp = np.eye(2, 3)
        multi_gmc(pool, warp)
        multi_gmc(unconfirmed, warp)
        matched, unmatched, remaining = associate(
            pool, high, self.args.match_thresh, self.args.fuse_score
        )
        activated = []
        for i, j in matched:
            pool[i].update(high[j], self.frame_id, self.kalman_filter)
            activated.append(pool[i])
        still_tracked = [pool[i] for i in unmatched if pool[i].state == TrackState.Tracked]
        matched, unmatched, _ = associate(still_tracked, low, 0.5)
        for i, j in matched:
            still_tracked[i].update(low[j], self.frame_id, self.kalman_filter)
            activated.append(still_tracked[i])
        newly_lost = [still_tracked[i] for i in unmatched]
        for track in newly_lost:
            track.state = TrackState.Lost
        remaining_high = [high[j] for j in remaining]
        matched, unmatched, remaining = associate(
            unconfirmed, remaining_high, 0.7, self.args.fuse_score
        )
        for i, j in matched:
            unconfirmed[i].update(remaining_high[j], self.frame_id, self.kalman_filter)
            activated.append(unconfirmed[i])
        removed = [unconfirmed[i] for i in unmatched]
        for j in remaining:
            self._next_id += 1
            track = remaining_high[j]
            track.activate(self.kalman_filter, self.frame_id, self._next_id)
            activated.append(track)
        # Expire before the next frame's association; removed IDs cannot be revived.
        lost = [t for t in self.lost_stracks + newly_lost if t.state == TrackState.Lost]
        expired = [t for t in lost if self.frame_id - t.end_frame > self.max_time_lost]
        removed.extend(expired)
        for track in removed:
            track.state = TrackState.Removed
        self.tracked_stracks = list({t.track_id: t for t in activated}.values())
        self.lost_stracks = list(
            {t.track_id: t for t in lost if t.state == TrackState.Lost}.values()
        )
        # Suppress duplicate recovered/new tracks as in BoT-SORT, keeping the older track.
        distances = iou_distance(self.tracked_stracks, self.lost_stracks)
        drop_active, drop_lost = set(), set()
        for i, j in zip(*np.where(distances < 0.15), strict=True):
            a, b = self.tracked_stracks[i], self.lost_stracks[j]
            if a.cls != b.cls:
                continue
            if a.frame_id - a.start_frame > b.frame_id - b.start_frame:
                drop_lost.add(j)
            else:
                drop_active.add(i)
        self.tracked_stracks = [
            t for i, t in enumerate(self.tracked_stracks) if i not in drop_active
        ]
        self.lost_stracks = [t for i, t in enumerate(self.lost_stracks) if i not in drop_lost]
        self.removed_stracks = (self.removed_stracks + removed)[-1000:]
        return np.asarray(
            [
                np.r_[t.xyxy, t.track_id, t.score, t.cls, t.index]
                for t in self.tracked_stracks
                if t.is_activated
            ],
            dtype=np.float32,
        ).reshape(-1, 8)
