# Table IV Reproduction Notes

- Best rows use the checkpoint with highest validation mRecall.
- Final rows use the last epoch.
- The paper reference is copied from the printed Table IV.
- Paper row `AutoTS w/o SIFT` has an mRecall inconsistency: the printed per-class recalls do not average to the printed 62.38.
- `AutoTS†` uses GT boxes for SIFT and localization points. ROI features use matched detector ROI features because this repo's DefaultPredictor exposes ROI vectors only for predicted boxes.
