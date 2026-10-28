"""Phase 2: fixed calibration / evaluation split of COCO val2017 vehicle images.

Calibration images are NEVER scored: if the activation ranges were fit on the eval images, mAP would be
measured on data the quantizer has already seen (the PTQ analogue of test-set leakage).
"""
import json, random, collections

VEHICLE = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}   # COCO category ids
N_CALIB, SEED = 150, 0

d = json.load(open("data/coco/annotations/instances_val2017.json"))
has_vehicle = {a["image_id"] for a in d["annotations"] if a["category_id"] in VEHICLE and not a["iscrowd"]}
ids = sorted(has_vehicle)                      # sort before shuffling: reproducible regardless of json order
random.Random(SEED).shuffle(ids)
files = {im["id"]: im["file_name"] for im in d["images"]}
split = {"seed": SEED, "calib": sorted(ids[:N_CALIB]), "eval": sorted(ids[N_CALIB:])}
split["files"] = {str(i): files[i] for i in ids}

cnt = collections.Counter(VEHICLE[a["category_id"]] for a in d["annotations"]
                          if a["image_id"] in set(split["eval"]) and a["category_id"] in VEHICLE and not a["iscrowd"])
json.dump(split, open("data/split.json", "w"))
print(f"calib={len(split['calib'])} eval={len(split['eval'])} eval instances={dict(cnt)}")
