__author__ = 'mikhail91'


from hough import Hough
from regression import TrackRegression
from metrics import HitsMatchingEfficiency
import numpy

class HoughClassification(Hough):

    def __init__(self, base=None, classifier=None, proba_threshold=0.8, track_eff_threshold=0.8):
        """
        This is modified Hough Transform method. This method uses tracks classification to reduce number of ghosts.

        Parameters
        ----------
        base : track pattern recognition method
            Base track pattern recignition method.
        classifier :  classifier
            Classifier for tracks classification.
        proba_threshold : float
            Predict probability threshold value.
        track_eff_threshold : float
            Track Efficiency threshold values to consider track as a good track.
        """

        self.base = base
        self.classifier = classifier
        self.proba_threshold = proba_threshold
        self.track_eff_threshold = track_eff_threshold
        self.min_hits = base.min_hits


    def new_track_inds(self, track_inds, X, classifier):
        """
        Reduces number of ghosts among recognized tracks using trained classifier.

        Parameters
        ----------
        track_inds : ndarray-like
            Array of recognized track indexes.
        X : ndarray-like
            Hit features.
        classifier : object
            Trained classifier.

        Returns
        -------
        new_track_inds : ndarray-like
            Array of new track indexes.
        """

        if classifier == None:
            return track_inds

        # Generate data for the classifier
        XX = []

        for track in track_inds:

            features = self.track_features(X[track,:])
            XX.append(features)

        XX = numpy.array(XX)


        # Predict probability to be a good track
        labels_pred = classifier.predict_proba(XX)[:, 1]


        # Select good tracks based on the probability
        track_inds = numpy.array(track_inds)
        new_track_inds = track_inds[labels_pred >= self.proba_threshold] # Larger the threshold value, better the ghosts reduction

        return new_track_inds

    def track_features(self, X):
        """
        Calculate track feature values for a classifier.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        n_hits : int
            Number of hits of a track.
        theta : float
            Theta parameter of a track.
        invr : float
            1/r0 parameter of a track.
        rmse : float
            RMSE of a track fit.
        """

        x, y = X[:, 3], X[:, 4]

        # Number of hits of a track
        n_hits = len(x)

        # Fit track parameters
        tr = TrackRegression()
        tr.fit(x, y)

        theta, invr = tr.theta_, tr.invr_


        # Predict hit coordinates of the fitted track
        phi = numpy.arctan(y / x) * (x != 0) + numpy.pi * (x < 0) + 0.5 * numpy.pi * (x==0) * (y>0) + 1.5 * numpy.pi * (x==0) * (y<0)
        x_pred, y_pred = tr.predict(phi)

        # Calculate RMSE
        rmse = numpy.sqrt(((y - y_pred)**2).sum())

        return n_hits, theta, invr, rmse

    def fit(self, X, y):
        """
        Train classifier to separate good track from ghost ones.

        Parameters
        ----------
        X : ndarray-like
            Hit features.
        y : array-like
            True hit labels.
        """

        XX = []
        yy = []

        # Create sample to train the classifier
        event_ids = numpy.unique(X[:, 0])

        for one_event_id in event_ids:

            # Select one event
            X_event = X[X[:, 0] == one_event_id]
            y_event = y[X[:, 0] == one_event_id]

            # The event track pattern recognition using the base method
            _ = self.base.predict_one_event(X_event)

            # Get recognized track inds. Approach: one hit can belong to several tracks
            track_inds = self.base.track_inds_

            # Calculate feature values for the recognized tracks
            for track in track_inds:

                # Select one recognized track
                X_track = X_event[track, :]
                y_track = y_event[track]

                # Calculate feature values for the track
                features = self.track_features(X_track)
                XX.append(features)

                # Calculate the track true labels in {0, 1}. 0 - ghost, 1 - good track.
                hme = HitsMatchingEfficiency(eff_threshold=self.track_eff_threshold, min_hits_per_track=self.min_hits)
                hme.fit(y_track, [1]*len(y_track))
                label = (hme.efficiencies_[0] >= self.track_eff_threshold) * 1.

                yy += [label]

        XX = numpy.array(XX)
        yy = numpy.array(yy)

        # Balance {0, 1} classes. This improves the classification quality.
        weights = numpy.zeros(len(yy))
        weights += 1. * len(yy) / len(yy[yy == 0]) * (yy == 0) + \
                   1. * len(yy) / len(yy[yy == 1]) * (yy == 1)

        # Train the classifier
        self.classifier.fit(XX, yy, weights)



    def predict_one_event(self, X):
        """
        Track patter recognition for one event.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        y : array-like
            Recognized track labels.
        """

        _ = self.base.predict_one_event(X)

        track_inds_ = self.new_track_inds(self.base.track_inds_, X, self.classifier)
        labels = self.get_hit_labels(track_inds_, len(X))

        return labels