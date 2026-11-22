"""Download and verify the Raster-to-Graph release checkpoint."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory


CHECKPOINT = Path(__file__).parent / "weights" / "checkpoint0299.pth"
RELEASE_URL = "https://drive.google.com/uc?id=1A_pyF0KHo-fja-fNqkoFkMUxo0W7GHhC"
RELEASE_SHA256 = "4fd7a69990da0aaa614ffd7a3c07537f0d4a3f072cd94684e857e23ebb7ccecb"


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    destination = parser.parse_args().output
    if destination.is_file() and digest(destination) == RELEASE_SHA256:
        print(f"Verified checkpoint: {destination}")
        return

    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / CHECKPOINT.name
        if not gdown.download(RELEASE_URL, str(temporary)):
            parser.exit(1, "Checkpoint download failed. See the setup instructions.\n")
        if digest(temporary) != RELEASE_SHA256:
            parser.exit(1, "Checkpoint checksum mismatch. The download was discarded.\n")
        temporary.replace(destination)
    print(f"Verified checkpoint: {destination}")


if __name__ == "__main__":
    main()
