import gdown
import os


urls_fns_dict = {
    "USPTO_50k_gln.mar": "https://drive.google.com/uc?id=1T0egYNi1tLZ2eOQuielJfGNvjrkdN4yb",
    "USPTO_50k_neuralsym.mar": "https://drive.google.com/uc?id=1NobcYvFc6KvH6EwbmXyYil6qpUjfr0Rx",
    "USPTO_full_neuralsym.mar": "https://drive.google.com/uc?id=1KOGFnMHGhA4BukXgh-D1APjA_yvkqygF",
    "pistachio_21Q1_neuralsym.mar": "https://drive.google.com/uc?id=15Gd6YgN9wkFjRG9M83GC8EL2e7LNlPkH",
    "USPTO_50k_transformer.mar": "https://drive.google.com/uc?id=1X6mvqmzEYxdXMZ_1C4xlGtPiSUngbKdp",
    "USPTO_full_transformer.mar": "https://drive.google.com/uc?id=1TTcd_woHZdyi4dtLQZzgZ3ZhkuuJ1XnC",
    "pistachio_21Q1_transformer.mar": "https://drive.google.com/uc?id=1_EVAd8DTmyUI0SRMF5DMhaVHnFDImxCo",
    "USPTO_50k_retroxpert.mar": "https://drive.google.com/uc?id=11biLTqrbKSJ4kDVwcFy35r6qHk0BncX2",
    "USPTO_full_retroxpert.mar": "https://drive.google.com/uc?id=1WW1o-XGur3tULxUmNPMfsS9xF-kHBmTc",
    "USPTO_50k_localretro.mar": "https://drive.google.com/uc?id=1zDpJ9GWsGZwerd8w74MbcwHfiRMqqiRR",
    "USPTO_50k_retrocomposer.mar": "https://drive.google.com/uc?id=11MbI-Pin-qEqgJLsZByuHY63erDwDGjV"
}


def main():
    for archive_name, url in urls_fns_dict.items():
        ofn = os.path.join("./mars", archive_name)
        if not os.path.exists(ofn):
            gdown.download(url, ofn, quiet=False)
            assert os.path.exists(ofn)
        else:
            print(f"{ofn} exists, skip downloading")


if __name__ == "__main__":
    main()
