import argparse
import random
from pathlib import Path
import pandas as pd
import re

def list_images(person_dir):
    return [p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]

def parse_imagenum(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0

def build_match_pairs(people, target):
    pairs, used = [], set()
    random.shuffle(people)

    for person in people:
        imgs = list_images(person)
        if len(imgs) < 2:
            continue

        max_pairs = min(target // len(people) + 1, len(imgs) * (len(imgs)-1) // 2)
        for _ in range(max_pairs):
            i1, i2 = random.sample(imgs, 2)
            key = tuple(sorted([i1.name, i2.name]))
            if key in used:
                continue
            used.add(key)

            pairs.append({
                "name": person.name,
                "imagenum1": parse_imagenum(i1.name),
                "imagenum2": parse_imagenum(i2.name)
            })

            if len(pairs) >= target:
                return pairs
    return pairs

def build_mismatch_pairs(people, target):
    pairs = []
    for _ in range(target * 3):
        p1, p2 = random.sample(people, 2)
        imgs1, imgs2 = list_images(p1), list_images(p2)
        if not imgs1 or not imgs2:
            continue

        i1, i2 = random.choice(imgs1), random.choice(imgs2)
        pairs.append({
            "name1": p1.name,
            "imagenum1": parse_imagenum(i1.name),
            "name2": p2.name,
            "imagenum2": parse_imagenum(i2.name)
        })

        if len(pairs) >= target:
            break
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--match_pairs", type=int, default=2000)
    parser.add_argument("--mismatch_pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    people = [p for p in args.data_dir.iterdir() if p.is_dir()]
    assert len(people) >= 2, "Need >= 2 identities"

    match = build_match_pairs(people, args.match_pairs)
    mismatch = build_mismatch_pairs(people, args.mismatch_pairs)

    out = Path("datasets")
    out.mkdir(exist_ok=True)

    pd.DataFrame(match).to_csv(out / "matchpairs_celeba.csv", index=False)
    pd.DataFrame(mismatch).to_csv(out / "mismatchpairs_celeba.csv", index=False)

    print(f"✓ Match pairs   : {len(match)}")
    print(f"✓ Mismatch pairs: {len(mismatch)}")

if __name__ == "__main__":
    main()
