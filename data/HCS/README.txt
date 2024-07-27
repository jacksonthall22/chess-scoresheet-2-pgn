The Handwritten Chess Scoresheet Dataset contains a set of single and double paged chess scoresheet images with ground truth labels for training and testing.

Images are named as follows:
[Game #]_pg[page #].png

Ground truth labels are formatted as follows:
[Game #]_[page #]_[move #]_[black/white] [ground truth]

Note: ground truth labels for testing - found in "testing_tags.txt" - do not include a page number as they are ground truths for the game represented by the corresponding two pages with the same game number. These labels are formatted as follows:
[Game #]_[move #]_[black/white] [ground truth]