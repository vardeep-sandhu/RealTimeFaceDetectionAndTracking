"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
from kalman_tracker import KalmanBoxTracker
from correlation_tracker import CorrelationTracker
from data_association import associate_detections_to_trackers


class Sort:

  def __init__(self, max_age=10, min_hits=5, use_dlib = False):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

    self.use_dlib = use_dlib

  def update(self, dets, embeddings):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    ## These are just predictions from the kalman tracker

    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict() #for kal!
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    ## Here matching these predictions with the detections and checking which one matches and which one does not
    if dets != []:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
          d = matched[np.where(matched[:,1]==t)[0],0]
          trk.update(dets[d,:][0], embeddings[d[0]]) ## for dlib re-intialize the trackers ?!
          
          
      #create and initialise new trackers for unmatched detections
      for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], embedding=embeddings[i])
        # trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        
        if dets == []:
          trk.update([])
        d = trk.get_state()
        
        if ((trk.hit_streak >= self.min_hits) or (trk.time_since_update < self.max_age)):
          ret.append(np.concatenate((d,[trk.label])).reshape(1,-1)) 
        
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          print("Face gone for too long. Looking for new face")
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))