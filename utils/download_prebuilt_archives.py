import gdown
import os


urls_fns_dict = {
    "USPTO_50k_gln.mar": "https://drive.google.com/uc?id=18bCEBinSNNgY4bReJW-a6xy7iCSZFQCr",
    "USPTO_50k_neuralsym.mar": "https://drive.google.com/uc?id=1sYxoAjemCJlPVCcTHVbUCnd6wnLyfWZf",
    "USPTO_full_neuralsym.mar": "https://drive.google.com/uc?id=1Xg7vNyHi8-okZYX4J8BcZ9oRYYRz7Q_A",
    "pistachio_21Q1_neuralsym.mar": "https://drive.google.com/uc?id=1xUNMYuzVUr1EYliGUzcHqYe5_3ESSZ4f",
    "USPTO_50k_transformer.mar": "https://drive.google.com/uc?id=1Sht80ADqb4OWv1Wt9rosHuGPni3jaTEl",
    "USPTO_full_transformer.mar": "https://drive.google.com/uc?id=1QOekovkUUPcNCG1aXNBhBP2u0TcOGFGZ",
    "pistachio_21Q1_transformer.mar": "https://drive.google.com/uc?id=1A8WMt56x9ATnxq_3nqI_6M9Z34zm5rIJ",
    "USPTO_50k_retroxpert.mar": "https://drive.google.com/uc?id=1O4rT6DnopcfT_6fFTg87f8wW-UckEV8a",
    "USPTO_full_retroxpert.mar": "https://drive.google.com/uc?id=1erq5PkmtWf7qvB5DB_AnxfTmNy3oDYwg",
    "USPTO_50k_localretro.mar": "https://drive.google.com/uc?id=1Izen7MYIGtXPqAzGG6qQrjOwTFE1txu0",
    "USPTO_50k_retrocomposer.mar": "https://drive.google.com/uc?id=1Q22g28TMSpd1y4gIxTXSGC9uy56vG2dm"
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
