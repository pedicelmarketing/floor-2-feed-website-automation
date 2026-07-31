"""
QA for generated architectural renders and walkthrough videos.

Two independent checks, deliberately measuring different things:

  depth_metrics.py  numeric -- does the generated video's apparent geometry track the
                    ground-truth geometry we rendered from the CAD file?
  gemini_judge.py   perceptual -- does it read as a physically coherent space in motion
                    (flicker, wobble, dreamlike parallax)?

Neither subsumes the other. The numeric check can pass on a uniformly mis-scaled room; the
perceptual check can pass on something that looks plausible but is not this building.
qa_runner.py combines them and treats a numeric failure as non-negotiable.
"""
