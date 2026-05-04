"""
Download model files from GitHub Release assets.

Models are too large for Git (239MB + 108MB + 27MB), so they are hosted
as GitHub Release assets and downloaded on first run.
"""

import os
import urllib.request

GITHUB_REPO = "Champion1102/Medguard"
RELEASE_TAG = "v1.0"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

FILES = {
    "models/baseline_svm.pkl": f"{BASE_URL}/baseline_svm.pkl",
    "models/densenet121_pathmnist.pth": f"{BASE_URL}/densenet121_pathmnist.pth",
    "models/hybrid_gmm.pkl": f"{BASE_URL}/hybrid_gmm.pkl",
    "models/resnet18_pathmnist.pth": f"{BASE_URL}/resnet18_pathmnist.pth",
    "data/pathmnist.npz": f"{BASE_URL}/pathmnist.npz",
}


def download_file(url, dest):
    """Download a file with progress indication."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  Done ({size_mb:.1f} MB)")


def ensure_assets():
    """Download any missing model/data files."""
    missing = {k: v for k, v in FILES.items() if not os.path.exists(k)}
    if not missing:
        return False

    print(f"Downloading {len(missing)} missing file(s) from GitHub Release...")
    for dest, url in missing.items():
        download_file(url, dest)
    print("All files ready.\n")
    return True


if __name__ == "__main__":
    ensure_assets()
